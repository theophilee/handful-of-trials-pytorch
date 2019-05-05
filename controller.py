from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader

#from torch.multiprocessing import Pool, set_start_method
#try:
#    set_start_method('spawn')
#except RuntimeError:
#    pass

from utils import *
from model import BootstrapEnsemble
from optimizer import CEMOptimizer


class MPC:
    def __init__(self, param_str, args):
        """Model predictive controller.

        Arguments:
            param_str (str): String descriptor of experiment hyper-parameters.
            args (DotMap): A DotMap of MPC parameters.
                .env (gym.env): Environment for which this controller will be used.
                .plan_hor (int): The planning horizon that will be used in optimization.
                .num_part (int): Number of particles used for propagation method.
                .batch_size (int): Batch size.
                .obs_preproc (func): A function which modifies observations before they
                    are passed into the model.
                .pred_postproc (func): A function which takes the previous observations
                    and model predictions and returns the input to the cost function on
                    observations.
                .targ_proc (func): A function which takes current observations and next
                    observations and returns the array of targetets (so that the model
                    learns the mapping obs -> targ_proc(obs, next_obs)).
                .get_cost (func): A function which computes the cost of a batch of
                    transitions.

                .model_cfg (DotMap): A DotMap of model parameters.
                    .ensemble_size (int): Number of bootstrap model.
                    .in_features (int): size of each input sample
                    .out_features (int): size of each output sample
                    .hid_features iterable(int): size of each hidden layer, can be empty
                    .activation: activation function, one of 'relu', 'swish'
                    .lr (float): learning rate for optimizer
                    .weight_decay (float): weight decay for model parameters

                .opt_cfg DotMap): A DotMap of optimizer parameters.
                    .iterations (int): The number of iterations to perform during
                        optimization.
                    .popsize (int): The number of candidate solutions to be sampled at
                        every iteration
                    .num_elites (int): The number of top solutions that will be used to
                        obtain the distribution at the next iteration.
        """
        self.env = args.env
        self.act_features = args.env.action_space.shape[0]
        self.obs_features = args.env.observation_space.shape[0]
        self.act_bound = args.env.action_space.high[0]

        self.ckpt_file = param_str + '_ckpt.pt'
        self.plan_hor = args.plan_hor
        self.num_part = args.num_part
        self.batch_size = args.batch_size
        self.num_nets = args.model_cfg.ensemble_size

        self.obs_preproc = args.obs_preproc
        self.pred_postproc = args.pred_postproc
        self.targ_proc = args.targ_proc
        self.get_cost = args.get_cost

        self.has_been_trained = False

        # Check arguments
        assert self.num_part % self.num_nets == 0

        # Action sequence optimizer
        self.optimizer = CEMOptimizer(args.env.action_space, self.plan_hor, **args.opt_cfg)

        # Dataset to train model
        self.X = torch.empty((0, args.model_cfg.in_features))
        self.Y = torch.empty((0, args.model_cfg.out_features))

        # Bootstrap ensemble model
        self.model = BootstrapEnsemble(**args.model_cfg)

    def train(self, obs, acts, train_split=0.9, iterative=True):
        self.has_been_trained = True

        # Preprocess new data
        X_new = torch.from_numpy(np.concatenate([self.obs_preproc(obs[:-1]), acts], axis=1)).float()
        Y_new = torch.from_numpy(self.targ_proc(obs[:-1], obs[1:])).float()

        if iterative:
            # Add new data to training set
            self.X = torch.cat((self.X, X_new))
            self.Y = torch.cat((self.Y, Y_new))
        else:
            self.X = X_new
            self.Y = Y_new

        # Store input statistics for normalization
        num_train = int(self.X.size(0) * train_split)
        self.model.fit_input_stats(self.X[:num_train])

        if iterative:
            # Compute per model mse and cross-entropy on new test data
            metrics = OrderedDict()
            metrics["model/mse/test"], metrics["model/xentropy/test"] = self.model.evaluate(
                X_new.to(TORCH_DEVICE), Y_new.to(TORCH_DEVICE))

        train_dataset = TensorDataset(self.X[:num_train], self.Y[:num_train])
        train_loader = DataLoader(train_dataset, batch_size=self.num_nets * self.batch_size, shuffle=True)
        test_dataset = TensorDataset(self.X[num_train:], self.Y[num_train:])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        train_mses, train_xentropies = [], []
        val_mses, val_xentropies = [], []
        early_stopping = EarlyStopping(ckpt_file=self.ckpt_file,
                                       patience=min(20, 10 * self.num_nets))

        # Training loop
        while not early_stopping.early_stop:
            train_mse = np.zeros((len(train_loader), self.num_nets))
            train_xentropy = np.zeros((len(train_loader), self.num_nets))
            val_mse = np.zeros((len(test_loader), self.num_nets))
            val_xentropy = np.zeros((len(test_loader), self.num_nets))

            for i, (X, Y) in enumerate(train_loader):
                X = X.view(self.num_nets, X.size(0) // self.num_nets, -1).to(TORCH_DEVICE)
                Y = Y.view(self.num_nets, Y.size(0) // self.num_nets, -1).to(TORCH_DEVICE)
                train_mse[i], train_xentropy[i] = self.model.update(X, Y)

            for i, (X, Y) in enumerate(test_loader):
                val_mse[i], val_xentropy[i] = self.model.evaluate(X.to(TORCH_DEVICE), Y.to(TORCH_DEVICE))

            train_mses.append(np.mean(train_mse, axis=0))
            train_xentropies.append(np.mean(train_xentropy, axis=0))
            val_mses.append(np.mean(val_mse, axis=0))
            val_xentropies.append(np.mean(val_xentropy, axis=0))

            # Stop if mean validation mse across all models stops decreasing
            early_stopping(np.mean(val_mse), self.model.net)

        # Load policy with best validation loss
        early_stopping.load_best(self.model.net)

        # Record train/val per model mse and cross-entropy for each epoch
        metrics["model/mse/train"] = train_mses
        metrics["model/xentropy/train"] = train_xentropies
        metrics["model/mse/val"] = val_mses
        metrics["model/xentropy/val"] = val_xentropies

        return metrics

    def act(self, obs):
        """Returns the action that this controller would take for a single observation obs.

        Arguments:
            obs (1D numpy.ndarray): Observation.

        Returns: Action (1D numpy.ndarray).
        """
        if not self.has_been_trained:
            return np.random.uniform(self.act_bound, -self.act_bound, (self.act_features,))

        return self.act_parallel(obs[np.newaxis])[0]

    def act_parallel(self, obs):
        """Returns the action that this controller would take for each of the observations
        in obs. Used to sample multiple rollouts in parallel.

        Arguments:
            obs (2D numpy.ndarray): Observations (num_obs, obs_features).

        Returns: Actions (2D numpy.ndarray).
        """
        plans = self.optimizer.obtain_solution(numpy_to_device(obs), self._compile_cost)
        # Return the first action of each plan
        return numpy_from_device(plans[:, 0])

    @torch.no_grad()
    def _compile_cost(self, plans, cur_obs):
        """Compute cost of plans (sequences of actions) starting at observations in
        cur_obs under the learned dynamics.

        Arguments:
            plans (3D torch.Tensor): Sequences of actions of shape
                (num_obs, num_plans, plan_hor * act_features).
            cur_obs (2D torch.Tensor): Starting observations to compile costs of shape
                (num_obs, obs_features).

       Returns:
           costs (2D torch.Tensor): Cost of plans of shape (num_obs, num_plans).
       """
        num_obs, num_plans = plans.shape[:2]

        # Reshape plans for parallel computation
        # 1 - (num_obs, num_plans, plan_hor * act_features)
        # 2 - (num_obs, num_plans, plan_hor, act_features)
        # 3 - (plan_hor, num_obs, num_plans, act_features)
        # 4 - (plan_hor, num_obs, num_plans, num_part, act_features)
        # 5 - (plan_hor, num_obs * num_plans * num_part, act_features)
        plans = plans.view(num_obs, num_plans, self.plan_hor, self.act_features)
        plans = plans.permute(2, 0, 1, 3)
        plans = plans.unsqueeze(-2).expand(-1, -1, -1, self.num_part, -1).contiguous()
        plans = plans.view(self.plan_hor, -1, self.act_features)

        # Reshape observations for parallel computation
        # 1 - (num_obs, obs_features)
        # 2 - (num_obs * num_plans * num_part, obs_features)
        obs = cur_obs.repeat(num_plans * self.num_part, 1)

        # Compute costs in parallel
        # Across starting observations, plans per observation and particles per plan
        costs = torch.zeros(num_obs, num_plans, self.num_part).to(TORCH_DEVICE)

        for t in range(self.plan_hor):
            acts = plans[t]

            # Predict next observation
            next_obs = self._predict_next_obs_divide(obs, acts)

            # Compute cost of transitions
            cost = self.get_cost(obs, acts, next_obs)
            costs += cost.view(num_obs, num_plans, self.num_part)

            obs = next_obs

        # Average cost over particles
        costs = costs.mean(dim=-1)

        return costs

    def _predict_next_obs_average(self, obs, acts):
        """Predict next observation by averaging predictions of all models in ensemble.

        Arguments:
            obs (2D torch.Tensor): Observations.

        Returns: Actions (2D torch.Tensor).
        """
        # Preprocess observations
        proc_obs = self.obs_preproc(obs)

        # Duplicate observations and actions to be processed by ensemble
        proc_obs = proc_obs.repeat(self.num_nets, 1, 1)
        acts = acts.repeat(self.num_nets, 1, 1)

        # Predict next observations
        mean, logvar = self.model.predict(torch.cat((proc_obs, acts), dim=-1))
        preds = mean + torch.randn_like(mean, device=TORCH_DEVICE) * logvar.exp().sqrt()

        # Average ensemble predictions
        avg_pred = preds.mean(dim=0)

        # Postprocess predictions
        return self.pred_postproc(obs, avg_pred)

    def _predict_next_obs_divide(self, obs, acts):
        """Predict next observation by dividing predictions among models in ensemble.

        Arguments:
            obs (2D torch.Tensor): Observations.

        Returns: Actions (2D torch.Tensor).
        """
        # Preprocess observations
        proc_obs = self.obs_preproc(obs)

        # Divide particles among models in ensemble
        proc_obs = self._to_bootstrap_shape(proc_obs)
        acts = self._to_bootstrap_shape(acts)

        # Predict next observations
        mean, logvar = self.model.predict(torch.cat((proc_obs, acts), dim=-1))
        preds = mean + torch.randn_like(mean, device=TORCH_DEVICE) * logvar.exp().sqrt()

        # Postprocess predictions
        preds = self._from_bootstrap_shape(preds)
        return self.pred_postproc(obs, preds)

    def _to_bootstrap_shape(self, input):
        # Reshape matrix to be processed by bootstrap ensemble model
        # 1 - (x * num_part, num_features)
        # 2 - (x, num_nets, num_parts / num_nets, num_features)
        # 3 - (num_nets, x * (num_parts / num_nets), num_features)
        num_features = input.shape[-1]
        reshaped = input.view(-1, self.num_nets, self.num_part // self.num_nets, num_features)
        reshaped = reshaped.transpose(0, 1).contiguous().view(self.num_nets, -1, num_features)
        return reshaped

    def _from_bootstrap_shape(self, input):
        # Reshape 3D tensor processed by bootstrap ensemble model to matrix
        # 1 - (num_nets, x * (num_parts / num_nets), num_features)
        # 2 - (num_nets, x, num_part / num_nets, num_features)
        # 3 - (x * num_part, num_features)
        num_features = input.shape[-1]
        reshaped = input.view(self.num_nets, -1, self.num_part // self.num_nets, num_features)
        reshaped = reshaped.transpose(0, 1).contiguous().view(-1, num_features)
        return reshaped