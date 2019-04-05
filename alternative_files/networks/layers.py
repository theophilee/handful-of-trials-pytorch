import torch
import torch.nn as nn
import numpy as np

from .utils import truncated_normal


class BootstrapLinear(nn.Module):
    def __init__(self, ensemble_size, in_features, out_features):
        """Applies a linear transformation to the incoming data.

        Args:
            ensemble_size (int): size of the bootstrap ensemble
            in_features (int): size of each input sample
            out_features (int): size of each output sample

        Shape:
            - Input: (ensemble_size, batch_size, in_features)
            - Output: (ensemble_size, batch_size, out_features)

        Attributes:
            weight: the learnable weights of the module of shape
                (ensemble_size, in_features, out_features) initialized
                as truncated normal
            bias: the learnable bias of the module of shape
                (ensemble_size, out_features) initialized as zero
        """
        super().__init__()
        self.weight = nn.Parameter(truncated_normal(size=(ensemble_size, in_features, out_features),
                                                    std=1.0 / (2.0 * np.sqrt(in_features))))
        self.bias = nn.Parameter(torch.zeros([ensemble_size, 1, out_features], dtype=torch.float32))

    def forward(self, input):
        return input.matmul(self.weight) + self.bias


class BootstrapGaussian(nn.Module):
    def __init__(self, ensemble_size, in_features, out_features):
        """Applies a linear transformation to the incoming data and outputs
        mean and diagonal log variance parameters of a Multivariate Gaussian.

        Args:
            ensemble_size (int): size of the bootstrap ensemble
            in_features (int): size of each input sample
            out_features (int): size of each output sample

        Shape:
            - Input: (ensemble_size, batch_size, in_features)
            - Output mean: (ensemble_size, batch_size, out_features)
            - Output logvar: (ensemble_size, batch_size, out_features)

        Attributes:
            lin: BootstrapLinear layer
            max_logvar: the learnable maximum log variance
            min_logvar: the learnable minimum log variance
        """
        super().__init__()
        self.out_features = out_features
        self.lin = BootstrapLinear(ensemble_size, in_features, out_features * 2)
        self.max_logvar = nn.Parameter(torch.ones(1, out_features, dtype=torch.float32) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, out_features, dtype=torch.float32) * 10.0)

    def forward(self, input):
        x = self.lin(input)
        mean = x[:, :, :self.out_features]
        logvar = x[:, :, self.out_features:]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar


class Swish(nn.Module):
    def forward(self, input):
        return (input * torch.sigmoid(input))