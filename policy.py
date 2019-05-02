import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict

from model.layers import Swish
from utils import *


ACTIVATIONS = {'relu': nn.ReLU(), 'swish': Swish(), 'tanh':nn.Tanh()}


class Policy:
    def __init__(self, param_str, env, hid_features, activation, batch_size, lr, weight_decay):
        """Parameterized reactive policy.

        Arguments:
            param_str (str): String descriptor of experiment hyper-parameters.
            env (gym.env): Environment for which this policy will be used.
            hid_features (int list): size of each hidden layer, can be empty
            activation: activation function, one of 'relu', 'swish', 'tanh'
            batch_size (int): Batch size.
            lr (float): Learning rate for optimizer.
            weight_decay (float): Weight decay for policy parameters.
        """
        # TODO add observation pre-processing as for model?
        # TODO add output non-linearity?
        self.ckpt_file = param_str + '_ckpt.pt'
        self.obs_features = env.observation_space.shape[0]
        self.act_features = env.action_space.shape[0]
        self.act_high, self.act_low = env.action_space.high, env.action_space.low
        self.batch_size = batch_size

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
        X_new = torch.from_numpy(obs).float()
        Y_new = torch.from_numpy(acts).float()

        if iterative:
            # Add new data to training set
            self.X = torch.cat((self.X, X_new))
            self.Y = torch.cat((self.Y, Y_new))
        else:
            self.X = X_new
            self.Y = Y_new

        # Compute input statistics for normalization
        num_train = int(self.X.size(0) * train_split)
        self._fit_input_stats(self.X[:num_train])

        if iterative:
            # Compute mse on new test data
            test_mse = self.criterion(self.predict(X_new.to(TORCH_DEVICE)),
                                      Y_new.to(TORCH_DEVICE)).item()

        train_dataset = TensorDataset(self.X[:num_train], self.Y[:num_train])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = TensorDataset(self.X[num_train:], self.Y[num_train:])
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        train_mses, val_mses = [], []
        early_stopping = EarlyStopping(ckpt_file=self.ckpt_file, patience=10)

        # Training loop
        while not early_stopping.early_stop:
            train_mse = np.zeros(len(train_loader))
            val_mse = np.zeros(len(test_loader))

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

            train_mses.append(np.mean(train_mse))
            val_mses.append(np.mean(val_mse))

            # Stop if validation error stops decreasing
            early_stopping(np.mean(val_mse), self.net)

        # Load policy with best validation loss
        early_stopping.load_best(self.net)

        # Record train/val/test MSE for each epoch
        metrics = OrderedDict()
        metrics["policy/mse/train"] = train_mses
        metrics["policy/mse/val"] = val_mses
        if iterative:
            metrics["policy/mse/test"] = test_mse

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