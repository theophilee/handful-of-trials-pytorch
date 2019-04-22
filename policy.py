import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from collections import OrderedDict

from model.layers import Swish


ACTIVATIONS = {'relu': nn.ReLU(), 'swish': Swish(), 'tanh':nn.Tanh()}
TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def numpy_to_device(arr):
    return torch.from_numpy(arr).float().to(TORCH_DEVICE)

def numpy_from_device(tensor):
    return tensor.cpu().numpy()


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
        # TODO might need to add an output activation
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

    def train(self, obs, acts):
        new_X = torch.from_numpy(obs).float()
        new_Y = torch.from_numpy(acts).float()

        # Add new data to training set
        self.X = torch.cat((self.X, new_X))
        self.Y = torch.cat((self.Y, new_Y))

        # Compute input statistics for normalization
        self._fit_input_stats(self.X)

        # Record MSE on new data (test set)
        metrics = OrderedDict()
        metrics["policy/mse/test"] = self.criterion(self.net(new_X.to(TORCH_DEVICE)),
                                                    new_Y.to(TORCH_DEVICE)).item()

        dataset = TensorDataset(self.X, self.Y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        num_batches = len(loader)

        train_mses = []
        for e in range(self.train_epochs):
            mse = 0

            for (X, Y) in loader:
                loss = self.criterion(self.net(X.to(TORCH_DEVICE)), Y.to(TORCH_DEVICE))
                mse += loss.item()

                # Take a gradient step
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            train_mses.append(mse / len(loader))

        # Record MSE on training set for each epoch
        metrics["policy/mse/train"] = train_mses
        #for i, mse in enumerate(train_mses):
        #    metrics["policy/mse/train/epoch_{}".format(i + 1)] = mse

        return metrics

    def act(self, obs):
        """Returns the action that this policy would take given observation obs.

        Arguments:
            obs (1D numpy.ndarray): The current observation.

        Returns: An action (1D numpy.ndarray).
        """
        with torch.no_grad():
            return numpy_from_device(self.net(numpy_to_device(obs)))

    def _fit_input_stats(self, input):
        # Store data statistics for normalization
        # TODO try input normalization and see if makes a difference
        self.input_mean = torch.mean(input, dim=0, keepdim=True).to(TORCH_DEVICE)
        self.input_std = torch.std(input, dim=0, keepdim=True).to(TORCH_DEVICE)