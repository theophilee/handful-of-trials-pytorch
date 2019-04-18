import torch
import torch.nn as nn

from .layers import BootstrapLinear, BootstrapGaussian, Swish


ACTIVATIONS = {'relu': nn.ReLU(), 'swish': Swish(), 'tanh': nn.Tanh()}
TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class BootstrapEnsemble:
    def __init__(self, ensemble_size, in_features, out_features, hid_features, activation,
                 lr, weight_decay):
        """ Ensemble of bootstrap model.

        Args:
            ensemble_size (int): size of the bootstrap ensemble
            in_features (int): size of each input sample
            out_features (int): size of each output sample
            hid_features (int list): size of each hidden layer, can be empty
            activation: activation function, one of 'relu', 'swish', 'tanh'
            lr (float): learning rate for optimizer
            weight_decay (float): weight decay for model parameters
        """
        self.ensemble_size = ensemble_size

        self.net = self._make_network(
            ensemble_size, in_features, out_features, hid_features, activation).to(TORCH_DEVICE)

        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

    def _make_network(self, ensemble_size, in_features, out_features, hid_features, activation):
        if len(hid_features) > 0:
            layers = []

            for in_f, out_f in zip([in_features] + hid_features, hid_features):
                layers.append(BootstrapLinear(ensemble_size, in_f, out_f))
                layers.append(ACTIVATIONS[activation])

            layers.append(BootstrapGaussian(ensemble_size, hid_features[-1], out_features))
            return nn.Sequential(*layers)

        else:
            nn.Sequential(BootstrapGaussian(ensemble_size, in_features, out_features))

    def fit_input_stats(self, input):
        # Store data statistics for normalization
        # TODO how important is input normalization?
        self.input_mean = torch.mean(input, dim=0, keepdim=True)
        self.input_std = torch.std(input, dim=0, keepdim=True)
        self.input_std.data[self.input_std.data < 1e-12] = 1.0

    def predict(self, input):
        # Normalize input
        input = (input - self.input_mean) / self.input_std
        return self.net(input)

    def update(self, input, targ):
        # Compute model predictions
        mean, logvar = self.predict(input)

        # Compute cross-entropy loss
        inv_var = torch.exp(-logvar)
        mse = (mean - targ) ** 2
        xentropy = mse * inv_var + logvar
        loss = xentropy.mean()

        # Special regularization for max and min log variance parameters
        # TODO is special regularization for max and min log variance parameters important?
        loss += 0.01 * (self.net[-1].max_logvar.sum() - self.net[-1].min_logvar.sum())

        # Take a gradient step
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Compute model-wise mean squared error and cross-entropy for diagnostics
        mses = mse.mean((-2, -1)).detach().cpu().numpy()
        xentropies = xentropy.mean((-2, -1)).detach().cpu().numpy()

        return mses, xentropies

    def evaluate(self, input, targ):
        mean, logvar = self.predict(input)

        inv_var = torch.exp(-logvar)
        mse = (mean - targ) ** 2
        xentropy = mse * inv_var + logvar

        mses = mse.mean((-2, -1)).detach().cpu().numpy()
        xentropies = xentropy.mean((-2, -1)).detach().cpu().numpy()

        return mses, xentropies