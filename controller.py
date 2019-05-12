from torch.utils.data import TensorDataset, DataLoader
from functools import partial
import time

#from torch.multiprocessing import Pool, set_start_method
#try:
#    set_start_method('spawn')
#except RuntimeError:
#    pass

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
                    .in_features (int): Size of each input sample.
                    .out_features (int): Size of each output sample.
                    .hid_features iterable(int): Size of each hidden layer, can be empty.
                    .activation: Activation function, one of 'relu', 'swish'.
                    .lr (float): Learning rate for optimizer.
                    .weight_decay (float): Weight decay for model parameters.

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
        self.reset_model = partial(self._reset_model, model_cfg=args.model_cfg)
        self.model = self.reset_model()

    def _reset_model(self, model_cfg):
        return BootstrapEnsemble(**model_cfg)

    def train(self, obs, acts, train_split=0.9, iterative=True, reset_model=False,
              debug_logger=None):
        """ Train bootstrap ensemble model.

        Arguments:
            obs (2D or 3D np.ndarray): observations
            acts (2D or 3D np.ndarray): actions
            train_split (float): proportion of data used for training
            iterative (bool): if True, add new data to training set otherwise
                start training set from scratch
            reset_model (bool): if True, reset model weights and optimizer
            debug_logger: if not None, plot metrics every epoch
        """
        self.has_been_trained = True

        if reset_model:
            self.model = self.reset_model()

        # Preprocess new data
        assert (obs.ndim in [2, 3]) and obs.ndim == acts.ndim
        if obs.ndim == 2:
            X_new = np.concatenate([self.obs_preproc(obs[:-1]), acts], axis=1)
            Y_new = self.targ_proc(obs[:-1], obs[1:])
        elif obs.ndim == 3:
            X_new = np.concatenate([np.concatenate([self.obs_preproc(o[:-1]), a], axis=1)
                                    for o, a in zip(obs, acts)], axis=0)
            Y_new = np.concatenate([self.targ_proc(o[:-1], o[1:]) for o in obs], axis=0)
        X_new, Y_new = torch.from_numpy(X_new).float(), torch.from_numpy(Y_new).float()

        # Add new data to training set
        if iterative:
            self.X, self.Y = torch.cat((self.X, X_new)), torch.cat((self.Y, Y_new))
        else:
            self.X, self.Y = X_new, Y_new

        # Store input statistics for normalization
        self.model.fit_input_stats(self.X)

        # Create bootstrap ensemble train-val splits
        dataset = TensorDataset(self.X, self.Y)
        train_size = int(train_split * len(self.X))
        val_size = len(self.X) - train_size
        train_loaders, val_loaders = [], []
        for _ in range(self.num_nets):
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_loaders.append(DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True))
            val_loaders.append(DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True))

        # Record mse and cross-entropy on new test data
        metrics = {}
        if iterative:
            mse, xentropy = self.model.evaluate(X_new.to(TORCH_DEVICE), Y_new.to(TORCH_DEVICE))
            metrics["model/mse/test"], metrics["model/xentropy/test"] = mse.mean(), xentropy.mean()

        early_stopping = EarlyStopping(patience=12)
        start = time.time()

        # Training loop
        epoch = 0
        while not early_stopping.early_stop:
            epoch += 1
            train_mse = torch.zeros((len(train_loaders[0]), self.num_nets))
            train_xentropy = np.zeros((len(train_loaders[0]), self.num_nets))
            val_mse = torch.zeros((len(val_loaders[0]), self.num_nets))
            val_xentropy = torch.zeros((len(val_loaders[0]), self.num_nets))

            for i, batch in enumerate(zip(*train_loaders)):
                X, Y = [torch.stack(b).to(TORCH_DEVICE) for b in zip(*batch)]
                train_mse[i], train_xentropy[i] = self.model.update(X, Y)

            for i, batch in enumerate(zip(*val_loaders)):
                X, Y = [torch.stack(b).to(TORCH_DEVICE) for b in zip(*batch)]
                val_mse[i], val_xentropy[i] = self.model.evaluate(X, Y)

            # Record epoch train/val mse and cross-entropy averaged across models
            info_step = {"metrics": {"model/mse/train": train_mse.mean(),
                                     "model/xentropy/train": train_xentropy.mean(),
                                     "model/mse/val": val_mse.mean(),
                                     "model/xentropy/val": val_xentropy.mean()},
                         "tensors": {"model/max_logvar": self.model.net[-1].max_logvar,
                                     "model/min_logvar": self.model.net[-1].min_logvar}}

            if debug_logger is not None:
                for k, v in info_step["metrics"].items():
                    debug_logger.log_scalar(k, v, epoch)
                for k, v in info_step["tensors"].items():
                    debug_logger.log_histogram(k, v, epoch)

            # Stop if mean validation cross-entropy across all models stops decreasing
            early_stopping.step(val_xentropy.mean(), self.model.net, info_step)

        # Load policy with best validation loss
        info_best = early_stopping.load_best(self.model.net)
        metrics.update(info_best["metrics"])
        metrics["model/train_time"] = time.time() - start

        return metrics, info_best["tensors"]

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

        # Predict next observations by averaging ensemble predictions
        proc_obs = proc_obs.repeat(self.num_nets, 1, 1)
        acts = acts.repeat(self.num_nets, 1, 1)
        preds = self.model.sample(torch.cat((proc_obs, acts), dim=-1))
        avg_preds = preds.mean(dim=0)

        # Postprocess predictions
        return self.pred_postproc(obs, avg_preds)

    def _predict_next_obs_divide(self, obs, acts):
        """Predict next observation by dividing predictions among models in ensemble.

        Arguments:
            obs (2D torch.Tensor): Observations.

        Returns: Actions (2D torch.Tensor).
        """
        # Preprocess observations
        proc_obs = self.obs_preproc(obs)

        # Predict next observations, by dividing particles among models in ensemble
        input = self._to_bootstrap_shape(torch.cat((proc_obs, acts), dim=-1))
        preds = self.model.sample(input)
        preds = self._from_bootstrap_shape(preds)

        # Postprocess predictions
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