from torch.utils.data import TensorDataset
from functools import partial
import time
import math

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
                .batches_per_epoch (int): Number of batches per training epoch.
                .obs_preproc (func): A function which modifies observations before they
                    are passed into the model.
                .pred_postproc (func): A function which takes the previous observations
                    and model predictions and returns the next observation.
                .targ_proc (func): A function which takes current observations and next
                    observations and returns the array of targetets (so that the model
                    learns the mapping obs -> targ_proc(obs, next_obs)).
                .get_reward (func): A function which computes the reward of a batch of
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
                    .iterations (int): The number of iterations to perform during CEM
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
        self.batches_per_epoch = args.batches_per_epoch
        self.num_nets = args.model_cfg.ensemble_size

        self.obs_preproc = args.obs_preproc
        self.pred_postproc = args.pred_postproc
        self.targ_proc = args.targ_proc
        self.get_reward = args.get_reward

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

    def train(self, obs, acts, train_split=0.8, iterative=True, reset_model=False,
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
            X_new = [np.concatenate([self.obs_preproc(o[:-1]), a], axis=1) for o, a in zip(obs, acts)]
            Y_new = [self.targ_proc(o[:-1], o[1:]) for o in obs]
            X_new, Y_new = np.concatenate(X_new, axis=0), np.concatenate(Y_new, axis=0)
        X_new, Y_new = torch.from_numpy(X_new).float(), torch.from_numpy(Y_new).float()

        # Add new data to training set
        if iterative:
            self.X, self.Y = torch.cat((self.X, X_new)), torch.cat((self.Y, Y_new))
        else:
            self.X, self.Y = X_new, Y_new

        # Store input statistics for normalization
        self.model.fit_input_stats(self.X)

        # Record mse and cross-entropy on new test data
        metrics = {}
        if iterative:
            mse, xentropy = self.model.evaluate(X_new.to(TORCH_DEVICE), Y_new.to(TORCH_DEVICE))
            metrics["model/mse/test"], metrics["model/xentropy/test"] = mse.mean(), xentropy.mean()

        dataset = TensorDataset(self.X, self.Y)
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size

        # Bootstrap ensemble train and validation indexes
        idxs = [torch.randperm(len(dataset)) for _ in range(self.num_nets)]
        train_idxs = torch.stack([i[:train_size] for i in idxs])
        val_idxs = torch.stack([i[train_size:] for i in idxs])

        batch_size = int(len(dataset) / self.batches_per_epoch)
        train_batches = int(train_split * self.batches_per_epoch)
        val_batches = self.batches_per_epoch - train_batches

        early_stopping = EarlyStopping(patience=20)
        start = time.time()

        # Training loop
        epoch = 0
        while not early_stopping.early_stop:
            epoch += 1
            train_idxs_epoch = train_idxs[:, torch.randperm(train_size)]
            val_idxs_epoch = val_idxs[:, torch.randperm(val_size)]

            train_mse = torch.zeros((train_batches, self.num_nets))
            train_xentropy = torch.zeros((train_batches, self.num_nets))
            train_reg = torch.zeros(train_batches)
            val_mse = torch.zeros((val_batches, self.num_nets))
            val_xentropy = torch.zeros((val_batches, self.num_nets))

            for i in range(train_batches):
                X, Y = dataset[train_idxs_epoch[:, i*batch_size:(i+1)*batch_size]]
                X, Y = X.to(TORCH_DEVICE), Y.to(TORCH_DEVICE)
                train_mse[i], train_xentropy[i], train_reg[i] = self.model.update(X, Y)

            for i in range(val_batches):
                X, Y = dataset[val_idxs_epoch[:, i*batch_size:(i+1)*batch_size]]
                X, Y = X.to(TORCH_DEVICE), Y.to(TORCH_DEVICE)
                val_mse[i], val_xentropy[i] = self.model.evaluate(X, Y)

            # Record epoch train/val mse and cross-entropy averaged across models
            info_step = {"metrics": {"model/mse/train": train_mse.mean(),
                                     "model/xentropy/train": train_xentropy.mean(),
                                     "model/regularization/train": train_reg.mean(),
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
        metrics["model/train_epochs"] = epoch - early_stopping.patience

        return metrics, info_best["tensors"]

    def act(self, obs):
        """Returns the action that this controller would take for a single observation obs.

        Arguments:
            obs (1D numpy.ndarray): Observation.

        Returns: Action (1D numpy.ndarray).
        """
        if not self.has_been_trained:
            return np.random.uniform(-self.act_bound, self.act_bound, (self.act_features,))

        return self.act_parallel(torch.from_numpy(obs[np.newaxis]).float())[0].cpu().numpy()

    def act_parallel(self, obs):
        """Returns the action that this controller would take for each of the observations
        in obs. Used to sample multiple rollouts in parallel.

        Arguments:
            obs (2D torch.Tensor): Observations (num_obs, obs_features).

        Returns: Actions (2D torch.Tensor).
        """
        plans = self.optimizer.obtain_solution(obs, self._compile_score)
        # Return the first action of each plan
        return plans[:, 0]

    @torch.no_grad()
    def sample_imaginary_rollouts(self, num, actor):
        """Sample multiple rollouts generated by a given actor under the learned dynamics in
        parallel.

        Arguments:
            num (int): Number of rollouts to sample.
            actor : Must provide an act_parallel() function operating on 2D torch.Tensor.

        Returns:
            obs (3D torch.Tensor): Trajectories of observations.
            acts (3D torch.Tensor): Trajectories of actions.
            avg_score (float): Average score.
        """
        # TODO split predictions among models in ensemble instead of averaging?
        # We use the environment only for its start state distribution
        observations = [torch.tensor([self.env.reset() for _ in range(num)]).float()]
        actions, scores = [], torch.zeros(num).to(TORCH_DEVICE)

        for t in range(self.env.num_steps):
            actions.append(actor.act_parallel(observations[t]))
            obs, acts = observations[t].to(TORCH_DEVICE), actions[t].to(TORCH_DEVICE)
            next_obs = self._predict_next_obs_average(obs, acts)
            scores += self.get_reward(obs, acts, next_obs)
            observations.append(next_obs.cpu())

        return torch.cat(observations)[:-1], torch.cat(actions), scores.mean().item()

    @torch.no_grad()
    def _compile_score(self, plans, cur_obs, batch_size=20000):
        """Compute score of plans (sequences of actions) starting at observations in
        cur_obs under the learned dynamics.

        Arguments:
            plans (3D torch.Tensor): Sequences of actions of shape
                (num_obs, num_plans, plan_hor * act_features).
            cur_obs (2D torch.Tensor): Starting observations to compile scores of shape
                (num_obs, obs_features).
            batch_size (int: Batch size for parallel computation.

        Returns:
            scores (2D torch.Tensor): Score of plans of shape (num_obs, num_plans).
        """
        assert batch_size % self.num_part == 0
        num_obs, num_plans = plans.shape[:2]

        # Reshape plans for parallel computation
        # 1 - (num_obs, num_plans, plan_hor * act_features)
        # 2 - (num_obs, num_plans, plan_hor, act_features)
        # 3 - (num_obs, num_plans, num_part, plan_hor, act_features)
        # 4 - (num_obs * num_plans * num_part, plan_hor, act_features)
        plans = plans.view(num_obs, num_plans, self.plan_hor, self.act_features)
        plans = plans.unsqueeze(2).expand(-1, -1, self.num_part, -1, -1).contiguous()
        plans = plans.view(-1, self.plan_hor, self.act_features)

        # Reshape observations for parallel computation
        # 1 - (num_obs, obs_features)
        # 2 - (num_obs * num_plans * num_part, obs_features)
        obs = cur_obs.repeat(num_plans * self.num_part, 1)

        dataset = TensorDataset(obs, plans)
        num_batches = math.ceil(len(dataset) / batch_size)
        scores = torch.zeros(num_obs * num_plans * self.num_part).to(TORCH_DEVICE)

        # Compute scores in parallel
        # Across starting observations, plans per observation and particles per plan
        for i in range(num_batches):
            obs, plans = dataset[i * batch_size:(i+1) * batch_size]
            obs, plans = obs.to(TORCH_DEVICE), plans.to(TORCH_DEVICE)
            for t in range(self.plan_hor):
                acts = plans[:, t]
                next_obs = self._predict_next_obs_divide(obs, acts)
                scores[i * batch_size:(i+1) * batch_size] += self.get_reward(obs, acts, next_obs)
                obs = next_obs

        # Average score over particles
        scores = scores.view(num_obs, num_plans, self.num_part).mean(dim=-1)
        return scores.cpu()

    def _predict_next_obs_average(self, obs, acts):
        """Predict next observation by averaging predictions of all models in ensemble.

        Arguments:
            obs (2D torch.Tensor): Observations.

        Returns: Actions (2D torch.Tensor).
        """
        # Preprocess observations
        proc_obs = self.obs_preproc(obs)

        # Predict next observations by averaging ensemble predictions
        input = torch.cat((proc_obs, acts), dim=-1).repeat(self.num_nets, 1, 1)
        preds = self.model.sample(input)
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