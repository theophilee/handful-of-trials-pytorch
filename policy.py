import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from model.layers import Swish
from utils import *


ACTIVATIONS = {'relu': nn.ReLU(), 'swish': Swish(), 'tanh':nn.Tanh()}


class Policy:
    def __init__(self, env, obs_features, hid_features, activation, batch_size, lr,
                 weight_decay, obs_preproc):
        """Parameterized reactive policy.

        Arguments:
            env (gym.env): Environment for which this policy will be used.
            obs_features (int): Size of each post-processed observation.
            hid_features (int list): Size of each hidden layer, can be empty.
            activation: Activation function, one of 'relu', 'swish', 'tanh'.
            batch_size (int): Batch size.
            lr (float): Learning rate for optimizer.
            weight_decay (float): Weight decay for policy parameters.
            obs_preproc (func): A function which modifies observations before they
                are passed into the policy.
        """
        self.obs_features = obs_features
        self.act_features = env.action_space.shape[0]
        self.act_bound = env.action_space.high[0]
        self.batch_size = batch_size
        self.obs_preproc = obs_preproc

        self.net = self._make_network(self.obs_features, self.act_features,
                                      hid_features, activation).to(TORCH_DEVICE)

        self.optim = Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.has_been_trained = False

        # Dataset to train policy
        self.X = torch.empty((0, self.obs_features))
        self.Y = torch.empty((0, self.act_features))
    
    def _make_network(self, obs_features, act_features, hid_features, activation):
        if len(hid_features) > 0:
            layers = []

            for in_f, out_f in zip([obs_features] + hid_features, hid_features):
                layers.append(nn.Linear(in_f, out_f))
                layers.append(ACTIVATIONS[activation])

            layers.append(nn.Linear(hid_features[-1], act_features))
            return nn.Sequential(*layers)

        else:
            nn.Sequential(nn.Linear(obs_features, act_features))

    def reset_training_set(self):
        # Reset dataset used to train policy (start of inner loop iteration)
        self.X = torch.empty((0, self.obs_features))
        self.Y = torch.empty((0, self.act_features))

    def train(self, obs, acts, train_split=0.8, iterative=True):
        """ Train policy.

        Arguments:
            obs (2D np.ndarray or 2D torch.Tensor): observations.
            acts (2D np.ndarray or 2D torch.Tensor): actions.
            train_split (float): proportion of data used for training
            iterative (bool): if True, add new data to training set otherwise
                start training set from scratch
            reset_model (bool): if True, reset model weights and optimizer
            debug_logger: if not None, plot metrics every epoch
        """
        self.has_been_trained = True

        # Preprocess new data
        if isinstance(obs, np.ndarray):
            obs, acts = torch.from_numpy(obs).float(), torch.from_numpy(acts).float()
        X_new, Y_new = self.obs_preproc(obs), acts

        if iterative:
            # Add new data to training set
            self.X, self.Y = torch.cat((self.X, X_new)), torch.cat((self.Y, Y_new))
        else:
            self.X, self.Y = X_new, Y_new

        # Compute input statistics for normalization
        self._fit_input_stats(self.X)

        # Compute mse on new test data
        metrics = {}
        if iterative:
            metrics["policy/mse/test"] = self.criterion(self.predict(X_new.to(TORCH_DEVICE)),
                                                        Y_new.to(TORCH_DEVICE)).item()

        dataset = TensorDataset(self.X, self.Y)
        train_size = int(train_split * len(self.X))
        val_size = len(self.X) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        early_stopping = EarlyStopping(patience=20)

        # Training loop
        while not early_stopping.early_stop:
            train_mse = torch.zeros(len(train_loader))
            val_mse = torch.zeros(len(val_loader))

            for i, (X, Y) in enumerate(train_loader):
                loss = self.criterion(self.predict(X.to(TORCH_DEVICE)), Y.to(TORCH_DEVICE))
                train_mse[i] = loss.item()

                # Take a gradient step
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            for i, (X, Y) in enumerate(val_loader):
                loss = self.criterion(self.predict(X.to(TORCH_DEVICE)), Y.to(TORCH_DEVICE))
                val_mse[i] = loss.item()

            info = {"policy/mse/train": train_mse.mean(),
                    "policy/mse/val": val_mse.mean()}

            # Stop if validation error stops decreasing
            early_stopping.step(val_mse.mean(), self.net, info)

        # Load policy with best validation loss
        info_best = early_stopping.load_best(self.net)
        metrics.update(info_best)
        return metrics

    def predict(self, input):
        # Normalize input
        input = (input - self.input_mean) / self.input_std
        return self.net(input)

    @torch.no_grad()
    def act_parallel(self, obs):
        """Returns the action that this policy would take for each of the observations in obs.

        Arguments:
            obs (2D torch.Tensor): Observations (num_obs, obs_features) on CPU.

        Returns: Actions (2D torch.Tensor) on CPU.
        """
        if not self.has_been_trained:
            acts = torch.FloatTensor(obs.shape[0], self.act_features)
            acts = acts.uniform_(-self.act_bound, self.act_bound)
            return acts

        return self.predict(self.obs_preproc(obs.to(TORCH_DEVICE))).cpu()

    @torch.no_grad()
    def act(self, obs):
        """Returns the action that this policy would take given observation obs.

        Arguments:
            obs (1D numpy.ndarray): An observation.

        Returns: An action (1D numpy.ndarray).
        """
        obs = torch.from_numpy(obs[np.newaxis]).float()
        return self.act_parallel(obs)[0].numpy()

    def _fit_input_stats(self, input):
        # Store data statistics for normalization
        self.input_mean = torch.mean(input, dim=0, keepdim=True).to(TORCH_DEVICE)
        self.input_std = torch.std(input, dim=0, keepdim=True).to(TORCH_DEVICE)
        self.input_std.data[self.input_std.data < 1e-12] = 1.0