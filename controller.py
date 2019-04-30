import copy
from collections import OrderedDict
from functools import partial

from torch.multiprocessing import Pool, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from utils import *
from model import BootstrapEnsemble
from optimizer import CEMOptimizer


class MPC:
    def __init__(self, args):
        """Model predictive controller.

        Arguments:
            args (DotMap): A DotMap of MPC parameters.
                .env (gym.env): Environment for which this controller will be used.
                .plan_hor (int): The planning horizon that will be used in optimization.
                .num_part (int): Number of particles used for propagation method.
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
                .get_cost (func): A function which computes the cost of a batch of
                    transitions.
                .reset_fns (list[func]): A list of function to be called when MPC.reset()
                    is called, can be empty.

                .model_cfg (DotMap): A DotMap of model parameters.
                    .ensemble_size (int): Number of bootstrap model.
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
        self.env = args.env
        self.act_features = args.env.action_space.shape[0]
        self.obs_features = args.env.observation_space.shape[0]
        self.act_high, self.act_low = args.env.action_space.high, args.env.action_space.low
        self.plan_hor = args.plan_hor
        self.num_part = args.num_part
        self.train_epochs = args.train_epochs
        self.batch_size = args.batch_size
        self.num_nets = args.model_cfg.ensemble_size

        self.obs_preproc = args.obs_preproc
        self.pred_postproc = args.pred_postproc
        self.targ_proc = args.targ_proc
        self.get_cost = args.get_cost
        self.reset_fns = args.reset_fns

        # Check arguments
        assert self.num_part % self.num_nets == 0

        # Action sequence optimizer
        self.optimizer = CEMOptimizer(
            lower_bound=np.tile(self.act_low, self.plan_hor),
            upper_bound=np.tile(self.act_high, self.plan_hor),
            **args.opt_cfg
        )

        # Controller state variables
        self.has_been_trained = False
        self.prev_plan = None

        # Dataset to train model
        self.X = np.empty((0, args.model_cfg.in_features))
        self.Y = np.empty((0, args.model_cfg.out_features))

        # Bootstrap ensemble model
        self.model = BootstrapEnsemble(**args.model_cfg)

    def train(self, obs, acts):
        self.has_been_trained = True

        # Preprocess new data
        new_X = np.concatenate([self.obs_preproc(obs[:-1]), acts], axis=1)
        new_Y = self.targ_proc(obs[:-1], obs[1:])

        # Add new data to training set
        self.X = np.concatenate([self.X, new_X])
        self.Y = np.concatenate([self.Y, new_Y])

        # Store input statistics for normalization
        self.model.fit_input_stats(numpy_to_device(self.X))

        # Record per model MSE and cross-entropy on new data (test set)
        metrics = OrderedDict()
        metrics["model/mse/test"], metrics["model/xentropy/test"] = self.model.evaluate(
            numpy_to_device(new_X), numpy_to_device(new_Y))

        num_examples = self.X.shape[0]
        num_batches = int(np.ceil(num_examples / self.batch_size))
        idxs = np.random.randint(num_examples, size=[self.num_nets, num_examples])

        mses = np.zeros((self.train_epochs, num_batches, self.num_nets))
        xentropies = np.zeros((self.train_epochs, num_batches, self.num_nets))

        # Training loop
        for e in range(self.train_epochs):
            for b in range(num_batches):
                batch_idxs = idxs[:, b * self.batch_size: (b + 1) * self.batch_size]

                # Take a gradient step
                mses[e, b], xentropies[e, b] = self.model.update(
                    numpy_to_device(self.X[batch_idxs]),
                    numpy_to_device(self.Y[batch_idxs]))

            np.random.shuffle(idxs)

        # Record per model mean squared error and cross-entropy on training set
        metrics["model/mse/train"] = mses.mean(axis=(0, 1))
        metrics["model/xentropy/train"] = xentropies.mean(axis=(0, 1))

        # Record parameters and gradients
        weights, grads = OrderedDict(), OrderedDict()
        for name, param in self.model.net.named_parameters():
            weights[name] = numpy_from_device(param.clone())
            grads[name] = numpy_from_device(param.grad.clone())

        return metrics, weights, grads

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None.
        """
        self.prev_plan = np.zeros((1, self.plan_hor * self.act_features))
        for fn in self.reset_fns:
            fn()

    def act_true_dynamics(self, env):
        """Returns the action that this controller would take for the current observation of
        environment env, using copies of env to generate CEM rollouts.

        Arguments:
            env (gym.env): Gym environment.

        Returns: Action (1D numpy.ndarray).
        """
        # Compute action plan over time horizon
        cost_function = partial(self._compile_cost_true_dynamics, env=env)
        plan = self.optimizer.obtain_solution(self.prev_plan, cost_function)

        # Store plan to initialize next iteration
        self.prev_plan = np.concatenate([plan[:, self.act_features:],
                                         np.zeros((1, self.act_features))],
                                        axis=1)

        # Return the first action
        act = plan[0, :self.act_features]
        return act

    def act(self, obs):
        """Returns the action that this controller would take for a single observation obs.

        Arguments:
            obs (1D numpy.ndarray): Observation.

        Returns: Action (1D numpy.ndarray).
        """
        if not self.has_been_trained:
            return np.random.uniform(self.act_low, self.act_high, self.act_low.shape)

        return self.act_parallel(obs[np.newaxis])[0]

    def act_parallel(self, obs):
        """Returns the action that this controller would take for each of the observations
        in obs. Used to sample multiple rollouts in parallel.

        Arguments:
            obs (2D numpy.ndarray): Observations (num_obs, obs_features).

        Returns: Actions (2D numpy.ndarray).
        """
        # Compute action plan over time horizon
        cost_function = partial(self._compile_cost, cur_obs=obs)
        plan = self.optimizer.obtain_solution(self.prev_plan, cost_function)

        # Store plan to initialize next iteration
        self.prev_plan = np.concatenate([plan[:, self.act_features:],
                                         np.zeros((obs.shape[0], self.act_features))],
                                        axis=1)

        # Return the first action
        acts = plan[:, :self.act_features]
        return acts

    def label(self, obs):
        """Returns the action that this controller would take for each of the observations
        in obs. Compared to act_parallel(), this function is not designed to be called
        multiple times sequentially to generate a rollout, but only once to label
        observations taken by another policy with the actions that this controller would
        take. This function does not use self.prev_plan (plans are initialized from scratch
        and not stored for subsequent calls).

        Arguments:
            obs (2D numpy.ndarray): Observations (num_obs, obs_features).

        Returns: Actions (2D numpy.ndarray).
        """
        # Compute action plan over time horizon
        init_mean = np.zeros((obs.shape[0], self.plan_hor * self.act_features))
        cost_function = partial(self._compile_cost, cur_obs=obs)
        plan = self.optimizer.obtain_solution(init_mean, cost_function)

        # Return the first action
        acts = plan[:, :self.act_features]
        return acts

    @torch.no_grad()
    def sample_imaginary_rollouts(self, actor, task_hor, num_rollouts):
        """Sample multiple rollouts generated by a given actor under the learned dynamics in
        parallel.

        Argument:
            actor (str): One of 'mpc', 'policy'.
            num_rollouts (int): Number of rollouts to sample.
            task_hor (int): Length of rollouts.

        Returns:
            obs (3D numpy.ndarray): Trajectories of observations.
            acts (3D numpy.ndarray): Trajectories of actions.
        """
        assert actor in ['mpc', 'policy']

        # We use the environment only for its start state distribution
        O = [np.array([self.env.reset() for _ in range(num_rollouts)])]
        A = []

        if actor == 'mpc':
            self.prev_plan = np.zeros((num_rollouts, self.plan_hor * self.act_features))

        for t in range(task_hor):
            # Compute next actions
            if actor == 'mpc':
                # This operation is the main computational bottleneck
                A.append(self.act_parallel(O[t]))
            else:
                raise NotImplementedError

            # Predict next observations by averaging predictions of bootstrap ensemble model
            # TODO split rollouts among models in ensemble instead of averaging?
            # or use different model at every step?
            obs = self._predict_next_obs_average(numpy_to_device(O[t]),
                                                 numpy_to_device(A[t]))

            O.append(numpy_from_device(obs))

        return np.array(O)[:-1], np.array(A)

    @torch.no_grad()
    def _compile_cost(self, plans, cur_obs):
        """Compute cost of plans (sequences of actions) starting at observations in
        cur_obs under the learned dynamics.

       Argument:
            plans (3D numpy.ndarray): Sequences of actions of shape
                (num_obs, num_plans, plan_hor * act_features).
            cur_obs (2D numpy.ndarray): Starting observations to compile costs of shape
                (num_obs, obs_features).

       Returns:
           costs (2D numpy.ndarray): Cost of plans of shape (num_obs, num_plans).
       """
        # TODO transfer CEM optimizer to GPU?
        num_obs, num_plans = plans.shape[:2]

        # Reshape plans for parallel computation
        # 1 - (num_obs, num_plans, plan_hor * act_features)
        # 2 - (num_obs, num_plans, plan_hor, act_features)
        # 3 - (plan_hor, num_obs, num_plans, act_features)
        # 4 - (plan_hor, num_obs, num_plans, num_part, act_features)
        # 5 - (plan_hor, num_obs * num_plans * num_part, act_features)
        plans = numpy_to_device(plans)
        plans = plans.view(num_obs, num_plans, self.plan_hor, self.act_features)
        plans = plans.permute(2, 0, 1, 3)
        plans = plans.unsqueeze(-2).expand(-1, -1, -1, self.num_part, -1).contiguous()
        plans = plans.view(self.plan_hor, -1, self.act_features)

        # Reshape observations for parallel computation
        # 1 - (num_obs, obs_features)
        # 2 - (num_obs * num_plans * num_part, obs_features)
        obs = numpy_to_device(cur_obs)
        obs = obs.repeat(num_plans * self.num_part, 1)

        # Compute costs in parallel
        # Across starting observations, plans per observation and particles per plan
        costs = np.zeros((num_obs, num_plans, self.num_part))

        for t in range(self.plan_hor):
            acts = plans[t]

            # Predict next observation
            next_obs = self._predict_next_obs_divide(obs, acts)

            # Compute cost of transitions
            cost = self.get_cost(obs, acts, next_obs)
            cost = cost.view(num_obs, num_plans, self.num_part)
            costs += numpy_from_device(cost)

            obs = next_obs

        # Average cost over particles
        costs = costs.mean(axis=-1)

        # Replace nan with high cost
        #costs[costs != costs] = 1e6

        return costs

    def _compile_cost_true_dynamics(self, plans, env):
        """Compute cost of plans (sequences of actions) starting at current observation
        of environment env under the true dynamics, using copies of env to generate CEM
        rollouts.

        Argument:
            plans (3D numpy.ndarray): Sequences of actions of shape
               (1, num_plans, plan_hor * act_features).
            env (gym.env): Gym environment.

        Returns:
          costs (2D numpy.ndarray): Cost of plans of shape (1, num_plans).
        """
        # TODO use multiple particles to evaluate plans?
        plans = [p.reshape(self.plan_hor, self.act_features) for p in plans[0]]
        eval_fn = partial(self._eval_plan, env=env)

        # Evaluate plans
        # TODO parallellize plan evaluation with multi-threading?
        # TODO why so SLOW? how to get faster access to true dynamics?
        costs = [eval_fn(plan) for plan in plans]
        #pool = Pool(16)
        #costs = pool.map(eval_fn, plans)

        return np.array(costs)[np.newaxis]

    def _eval_plan(self, plan, env):
        """Compute cost of plan from current observation of env using true dynamics.
        """
        env_copy = copy.deepcopy(env)
        total_cost = 0
        for act in plan:
            _, reward, _, _ = env_copy.step(act)
            total_cost -= reward

        return total_cost

    def _predict_next_obs_average(self, obs, acts):
        """Predict next observation, this function is called when sampling rollouts under
        the model wih sample_imaginary_rollouts(), it averages predictions of models in
        ensemble.

        Arguments:
            obs (2D numpy.ndarray): Observations.

        Returns: Actions (2D numpy.ndarray).
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
        """Predict next observation, this function is called when we propagate particles
         with _compile_costs(), it divides predictions among models in the ensemble.

        Arguments:
            obs (2D numpy.ndarray): Observations.

        Returns: Actions (2D numpy.ndarray).
        """
        # Preprocess observations
        proc_obs = self.obs_preproc(obs)

        # Divide particles among models in ensemble
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
        # 1 - (x * num_part, num_features)
        # 2 - (x, num_nets, num_parts / num_nets, num_features)
        # 3 - (num_nets, x * (num_parts / num_nets), num_features)
        num_features = input.shape[-1]
        reshaped = input.view(-1, self.num_nets, self.num_part // self.num_nets, num_features)
        reshaped = reshaped.transpose(0, 1).contiguous().view(self.num_nets, -1, num_features)
        return reshaped

    def _to_2D(self, input):
        # Reshape 3D tensor processed by bootstrap ensemble model to matrix
        # 1 - (num_nets, x * (num_parts / num_nets), num_features)
        # 2 - (num_nets, x, num_part / num_nets, num_features)
        # 3 - (x * num_part, num_features)
        num_features = input.shape[-1]
        reshaped = input.view(self.num_nets, -1, self.num_part // self.num_nets, num_features)
        reshaped = reshaped.transpose(0, 1).contiguous().view(-1, num_features)
        return reshaped