import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict

from model.layers import Swish
from utils import *


ACTIVATIONS = {'relu': nn.ReLU(), 'swish': Swish(), 'tanh':nn.Tanh()}


class Policy:
    def __init__(self, env, hid_features, activation, train_epochs, batch_size,
                 lr, weight_decay):
        """Parameterized reactive policy.
            .env (gym.env): Environment for which this policy will be used.
            .hid_features (int list): size of each hidden layer, can be empty
            .activation: activation function, one of 'relu', 'swish', 'tanh'
            .train_epochs (int): Number of epochs of training each time we refit
                the policy.
            .batch_size (int): Batch size.
            .lr (float): Learning rate for optimizer.
            .weight_decay (float): Weight decay for policy parameters.
        """
        # TODO add observation pre-processing as for model?
        self.obs_features = env.observation_space.shape[0]
        self.act_features = env.action_space.shape[0]
        self.act_high, self.act_low = env.action_space.high, env.action_space.low
        self.train_epochs = train_epochs
        self.batch_size = batch_size

        self.net = self._make_network(self.obs_features, self.act_features,
                                      hid_features, activation).to(TORCH_DEVICE)

        self.optim = Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

        self.X = torch.empty((0, self.obs_features))
        self.Y = torch.empty((0, self.act_features))
    
    def _make_network(self, obs_features, act_features, hid_features, activation):
        # TODO might need to add an output activation?
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

    def train(self, obs, acts, train_split=0.8, iterative=False):
        if iterative:
            # Add new data to training set
            self.X = torch.cat((self.X, torch.from_numpy(obs).float()))
            self.Y = torch.cat((self.Y, torch.from_numpy(acts).float()))
        else:
            self.X = torch.from_numpy(obs).float()
            self.Y = torch.from_numpy(acts).float()

        # Compute input statistics for normalization
        num_train = int(self.X.size(0) * train_split)
        self._fit_input_stats(self.X[:num_train])

        train_dataset = TensorDataset(self.X[:num_train], self.Y[:num_train])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = TensorDataset(self.X[num_train:], self.Y[num_train:])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        train_mses, test_mses = [], []
        early_stopping = EarlyStopping()

        # Train until test error stops decreasing
        while not early_stopping.early_stop:
            train_mse = 0
            for (X, Y) in train_loader:
                loss = self.criterion(self.predict(X.to(TORCH_DEVICE)), Y.to(TORCH_DEVICE))
                train_mse += loss.item()

                # Take a gradient step
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            train_mses.append(train_mse / len(train_loader))

            test_mse = 0
            for (X, Y) in test_loader:
                loss = self.criterion(self.predict(X.to(TORCH_DEVICE)), Y.to(TORCH_DEVICE))
                test_mse += loss.item()

            test_mses.append(test_mse / len(test_loader))
            early_stopping(test_mse / len(test_loader), self.net)

        # Load policy with best validation loss
        early_stopping.load_best(self.net)

        # Record train/test MSE for each epoch
        metrics = OrderedDict()
        metrics["policy/mse/train"] = train_mses
        metrics["policy/mse/test"] = test_mses

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
        with torch.no_grad():
            return numpy_from_device(self.predict(numpy_to_device(obs)))

    def _fit_input_stats(self, input):
        # Store data statistics for normalization
        self.input_mean = torch.mean(input, dim=0, keepdim=True).to(TORCH_DEVICE)
        self.input_std = torch.std(input, dim=0, keepdim=True).to(TORCH_DEVICE)