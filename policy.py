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
        # TODO add observation pre-processing as for model?
        # TODO add output non-linearity?
        self.obs_features = obs_features
        self.act_features = env.action_space.shape[0]
        self.act_high, self.act_low = env.action_space.high, env.action_space.low
        self.batch_size = batch_size
        self.obs_preproc = obs_preproc

        self.net = self._make_network(self.obs_features, self.act_features,
                                      hid_features, activation).to(TORCH_DEVICE)

        self.optim = Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

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

    def train(self, obs, acts, train_split=0.9, iterative=True):
        X_new = torch.from_numpy(self.obs_preproc(obs)).float()
        Y_new = torch.from_numpy(acts).float()

        if iterative:
            # Add new data to training set
            self.X, self.Y = torch.cat((self.X, X_new)), torch.cat((self.Y, Y_new))
        else:
            self.X, self.Y = X_new, Y_new

        # Compute input statistics for normalization
        num_train = int(self.X.size(0) * train_split)
        self._fit_input_stats(self.X[:num_train])

        metrics = {}
        if iterative:
            # Compute mse on new test data
            metrics["policy/mse/test"] = self.criterion(self.predict(X_new.to(TORCH_DEVICE)),
                                                        Y_new.to(TORCH_DEVICE)).item()

        train_dataset = TensorDataset(self.X[:num_train], self.Y[:num_train])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = TensorDataset(self.X[num_train:], self.Y[num_train:])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        early_stopping = EarlyStopping(patience=10)

        # Training loop
        while not early_stopping.early_stop:
            train_mse = torch.zeros(len(train_loader))
            val_mse = torch.zeros(len(test_loader))

            for i, (X, Y) in enumerate(train_loader):
                loss = self.criterion(self.predict(X.to(TORCH_DEVICE)), Y.to(TORCH_DEVICE))
                train_mse[i] = loss.item()

                # Take a gradient step
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            for i, (X, Y) in enumerate(test_loader):
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

    def act(self, obs):
        """Returns the action that this policy would take given observation obs.

        Arguments:
            obs (1D numpy.ndarray): The current observation.

        Returns: An action (1D numpy.ndarray).
        """
        obs = self.obs_preproc(obs[np.newaxis])[0]
        with torch.no_grad():
            return numpy_from_device(self.predict(numpy_to_device(obs)))

    def _fit_input_stats(self, input):
        # Store data statistics for normalization
        self.input_mean = torch.mean(input, dim=0, keepdim=True).to(TORCH_DEVICE)
        self.input_std = torch.std(input, dim=0, keepdim=True).to(TORCH_DEVICE)
        self.input_std.data[self.input_std.data < 1e-12] = 1.0