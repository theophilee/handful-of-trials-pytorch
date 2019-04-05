import torch
import numpy as np
from sklearn.model_selection import train_test_split
from collections import OrderedDict

from networks import BootstrapEnsemble
from optimizer import CEMOptimizer


TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


def numpy_to_device(arr):
    return torch.from_numpy(arr).float().to(TORCH_DEVICE)


class MPC:
    def __init__(self, args):
        """Model predictive controller.

        Arguments:
            args (DotMap): A DotMap of MPC parameters.
                .env (gym.env): Environment for which this controller will be used.
                .plan_hor (int): The planning horizon that will be used in optimization.
                .num_part (int): Number of particles used for DS, TSinf, TS1, and MM
                    propagation methods.
                .train_epochs (int): Number of epochs of training each time we refit
                    the model.
                .batch_size (int): Batch size.
                .obs_preproc (func): A function which modifies observations before they
                    are passed into the model.
                .pred_postproc (func): A function which takes the previous observations
                    and model predictions and returns the input to the cost function on
                    observations.
                .targ_proc (func): A function which takes current observations and next
                    observations and returns the array of targetets (so that the model
                    learns the mapping obs -> targ_proc(obs, next_obs)).
                .get_cost_obs (func): A function which computes the cost of a batch of
                    observations.
                .get_cost_acts (func): A function which computes the cost of a batch of
                    actions.
                .reset_fns (list[func]): A list of function to be called when MPC.reset()
                    is called, can be empty.

                .model_cfg (DotMap): A DotMap of model parameters.
                    .ensemble_size (int): Number of bootstrap networks.
                    .in_features (int): size of each input sample
                    .out_features (int): size of each output sample
                    .hid_features iterable(int): size of each hidden layer, can be empty
                    .activation: activation function, one of 'relu', 'swish'
                    .lr (float): learning rate for optimizer
                    .weight_decay (float): weight decay for model parameters

                .opt_cfg DotMap): A DotMap of optimizer parameters.
                    .max_iters (int): The maximum number of iterations to perform during
                        optimization.
                    .popsize (int): The number of candidate solutions to be sampled at
                        every iteration
                    .num_elites (int): The number of top solutions that will be used to
                        obtain the distribution at the next iteration.
                    .epsilon (float): If the maximum variance drops below this threshold,
                        optimization is stopped.
                    .alpha (float): Controls how much of the previous mean and variance is
                        used for the next iteration.
        """
        self.act_features = args.env.action_space.shape[0]
        self.act_high, self.act_low = args.env.action_space.high, args.env.action_space.low
        self.plan_hor = args.plan_hor
        self.num_part = args.num_part
        self.train_epochs = args.train_epochs
        self.batch_size = args.batch_size
        self.num_nets = args.model_cfg.ensemble_size

        self.obs_preproc = args.obs_preproc
        self.pred_postproc = args.pred_postproc
        self.targ_proc = args.targ_proc
        self.get_cost_obs = args.get_cost_obs
        self.get_cost_acts = args.get_cost_acts
        self.reset_fns = args.reset_fns

        # Check arguments
        assert self.num_part % self.num_nets == 0

        # Action sequence optimizer
        self.optimizer = CEMOptimizer(
            sol_dim=self.plan_hor * self.act_features,
            lower_bound=np.tile(self.act_low, self.plan_hor),
            upper_bound=np.tile(self.act_high, self.plan_hor),
            cost_function=self._compile_cost,
            **args.opt_cfg
        )

        # Controller state variables
        self.has_been_trained = False
        self.prev_plan = np.zeros(self.plan_hor * self.act_features)
        self.init_var = np.tile(np.square(self.act_high - self.act_low) / 16, self.plan_hor)

        # Dataset to train model
        self.X = np.empty((0, args.model_cfg.in_features))
        self.Y = np.empty((0, args.model_cfg.out_features))

        # Bootstrap ensemble model
        self.model = BootstrapEnsemble(**args.model_cfg)
        # TODO add dataparallel and device_ids as arguments?

    def train(self, obs, acts):
        self.has_been_trained = True

        # Add new data to training set
        new_X = np.concatenate([self.obs_preproc(obs[:-1]), acts], axis=1)
        new_Y = self.targ_proc(obs[:-1], obs[1:])

        self.X = np.concatenate([self.X, new_X])
        self.Y = np.concatenate([self.Y, new_Y])

        # Store input statistics for normalization
        self.model.fit_input_stats(torch.from_numpy(self.X).float())

        # Use 10% of data for validation
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.1)

        num_train = X_train.shape[0]
        num_batches = int(np.ceil(num_train / self.batch_size))
        idxs = np.random.randint(num_train, size=[self.num_nets, num_train])

        train_mses = np.zeros((self.train_epochs, num_batches, self.num_nets))
        train_xentropies = np.zeros((self.train_epochs, num_batches, self.num_nets))

        # Training loop
        for e in range(self.train_epochs):
            for b in range(num_batches):
                batch_idxs = idxs[:, b * self.batch_size: (b + 1) * self.batch_size]

                # Take a gradient step
                train_mses[e, b], train_xentropies[e, b] = self.model.update(
                    numpy_to_device(X_train[batch_idxs]),
                    numpy_to_device(Y_train[batch_idxs]))

            np.random.shuffle(idxs)

        # Record per model mean squared error and cross-entropy
        metrics = OrderedDict()
        metrics["mse/train"] = train_mses.mean(axis=(0, 1))
        metrics["xentropy/train"] = train_xentropies.mean(axis=(0, 1))
        metrics["mse/test"], metrics["xentropy/test"] = self.model.evaluate(
            numpy_to_device(X_test), numpy_to_device(Y_test))

        # Record parameters and gradients
        weights, grads = OrderedDict(), OrderedDict()
        for name, param in self.model.net.named_parameters():
            weights[name] = param.clone().cpu().data.numpy()
            grads[name] = param.grad.clone().cpu().data.numpy()

        return metrics, weights, grads

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None.
        """
        self.prev_plan = np.zeros(self.plan_hor * self.act_features)
        for fn in self.reset_fns:
            fn()

    def act(self, obs):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation.

        Returns: An action (1D numppy.ndarray).
        """
        if not self.has_been_trained:
            return np.random.uniform(self.act_low, self.act_high, self.act_low.shape)

        # Store current observation for self._compile_cost()
        self.cur_obs = obs

        # Compute action plan over time horizon
        plan = self.optimizer.obtain_solution(self.prev_plan, self.init_var)

        # Store plan to initialize next iteration
        self.prev_plan = np.concatenate([plan[self.act_features:], np.zeros(self.act_features)])

        # Return the first action
        action = plan[:self.act_features]
        return action

    @torch.no_grad()
    def _compile_cost(self, plans):
        num_plans = plans.shape[0]

        # TODO transfer CEM optimizer to GPU?
        # Reshape plans for parallel compute
        # 1 - (num_plans, plan_hor * act_features)
        # 2 - (num_plans, plan_hor, act_features)
        # 3 - (plan_hor, num_plans, act_features)
        # 4 - (plan_hor, num_plans, num_part, act_features)
        # 5 - (plan_hor, num_plans * num_part, act_features)
        plans = numpy_to_device(plans)
        plans = plans.view(-1, self.plan_hor, self.act_features)
        plans = plans.transpose(0, 1)
        plans = plans.unsqueeze(-2).expand(-1, -1, self.num_part, -1)
        plans = plans.contiguous().view(self.plan_hor, -1, self.act_features)

        # Reshape current observation for parallel compute
        # 1 - (obs_features)
        # 2 - (num_plans * num_part, obs_features)
        obs = numpy_to_device(self.cur_obs)
        obs = obs.unsqueeze(0).expand(num_plans * self.num_part, -1)

        # Fill cost matrix
        costs = np.zeros((num_plans, self.num_part))

        for t in range(self.plan_hor):
            acts = plans[t]

            # Predict next observation
            next_obs = self._predict_next_obs(obs, acts)

            # Compute cost of transition
            cost = self.get_cost_obs(next_obs) + self.get_cost_acts(acts)
            cost = cost.view(num_plans, self.num_part)
            costs += cost.cpu().numpy()

            obs = next_obs

        # Replace nan with high cost
        costs[costs != costs] = 1e6

        return costs.mean(axis=1)

    def _predict_next_obs(self, obs, acts):
        # Preprocess observations
        proc_obs = self.obs_preproc(obs)

        # Reshape observations and actions
        proc_obs = self._to_3D(proc_obs)
        acts = self._to_3D(acts)

        # Predict next observations
        mean, logvar = self.model.predict(torch.cat((proc_obs, acts), dim=-1))
        preds = mean + torch.randn_like(mean, device=TORCH_DEVICE) * logvar.exp().sqrt()

        # Reshape predictions
        preds = self._to_2D(preds)

        # Postprocess predictions
        return self.pred_postproc(obs, preds)

    def _to_3D(self, input):
        # Reshape matrix to be processed by bootstrap ensemble model
        # 1 - (num_plans * num_part, num_features)
        # 2 - (num_plans, num_nets, num_parts / num_nets, num_features)
        # 3 - (num_nets, num_plans * (num_parts / num_nets), num_features)
        num_features = input.shape[-1]
        reshaped = input.view(-1, self.num_nets, self.num_part // self.num_nets, num_features)
        reshaped = reshaped.transpose(0, 1).contiguous().view(self.num_nets, -1, num_features)
        return reshaped

    def _to_2D(self, input):
        # Reshape 3D tensor processed by bootstrap ensemble model to matrix
        # 1 - (num_nets, num_plans * (num_parts / num_nets), num_features)
        # 2 - (num_nets, num_plans, num_part / num_nets, num_features)
        # 3 - (num_plans * num_part, num_features)
        num_features = input.shape[-1]
        reshaped = input.view(self.num_nets, -1, self.num_part // self.num_nets, num_features)
        reshaped = reshaped.transpose(0, 1).contiguous().view(-1, num_features)
        return reshaped
