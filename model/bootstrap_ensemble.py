from .layers import *


ACTIVATIONS = {'relu': nn.ReLU(), 'swish': Swish(), 'tanh': nn.Tanh()}
TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class BootstrapEnsemble:
    def __init__(self, stochasticity, ensemble_size, in_features, out_features, hid_features,
                 activation, lr, weight_decay):
        """ Ensemble of bootstrap model.

        Args:
            stochasticity (str): one of 'deterministic', 'gaussian', 'gaussian_bias'
            ensemble_size (int): size of the bootstrap ensemble
            in_features (int): size of each input sample
            out_features (int): size of each output sample
            hid_features (int list): size of each hidden layer, can be empty
            activation: activation function, one of 'relu', 'swish', 'tanh'
            lr (float): learning rate for optimizer
            weight_decay (float): weight decay for model parameters
        """
        self.stochasticity = stochasticity
        self.net = self._make_network(
            ensemble_size, in_features, out_features, hid_features, activation).to(TORCH_DEVICE)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

    def _make_network(self, ensemble_size, in_features, out_features, hid_features, activation):
        layers = []

        for in_f, out_f in zip([in_features] + hid_features, hid_features):
            layers.append(BootstrapLinear(ensemble_size, in_f, out_f))
            layers.append(ACTIVATIONS[activation])

        if self.stochasticity == 'deterministic':
            output = BootstrapLinear
        elif self.stochasticity == 'gaussian':
            output = BootstrapGaussian
        elif self.stochasticity == 'gaussian_bias':
            output = BootstrapGaussianBias
        layers.append(output(ensemble_size, hid_features[-1], out_features))

        return nn.Sequential(*layers)

    def fit_input_stats(self, input):
        # Store data statistics for normalization
        self.input_mean = torch.mean(input, dim=0, keepdim=True).to(TORCH_DEVICE)
        self.input_std = torch.std(input, dim=0, keepdim=True).to(TORCH_DEVICE)
        self.input_std.data[self.input_std.data < 1e-12] = 1.0

    def predict(self, input):
        # Normalize input
        input = (input - self.input_mean) / self.input_std
        return self.net(input)

    def sample(self, input):
        if self.stochasticity == 'deterministic':
            return self.predict(input)
        else:
            mean, logvar = self.predict(input)
            return mean + torch.randn_like(mean, device=TORCH_DEVICE) * logvar.exp().sqrt()

    def update(self, input, targ):
        metrics = {}

        if self.stochasticity == 'deterministic':
            mean = self.predict(input)

            # Mean squared error loss
            mse = (mean - targ) ** 2
            xentropy = mse # To share code with stochastic dynamics
            loss = mse.mean()

        else:
            mean, logvar = self.predict(input)

            # Cross-entropy loss
            inv_var = torch.exp(-logvar)
            mse = (mean - targ) ** 2
            xentropy = mse * inv_var + logvar
            loss = xentropy.mean()

            if self.stochasticity == 'gaussian':
                # Small special regularization for max and min log variance parameters
                reg = 1e-6 * (self.net[-1].max_logvar.mean() - self.net[-1].min_logvar.mean())
                loss += reg

            metrics['logvar/mean_train'] = logvar.mean()
            metrics['logvar/min_train'] = logvar.min()
            metrics['logvar/max_train'] = logvar.max()
            metrics['logvar/std_train'] = logvar.std()

        mse = mse.cpu().detach()
        xentropy = xentropy.cpu().detach()
        metrics['mse/mean_train'] = mse.mean()
        metrics['mse/min_train'] = mse.min()
        metrics['mse/max_train'] = mse.max()
        metrics['mse/std_train'] = mse.std()
        metrics['xentropy/mean_train'] = xentropy.mean()

        # Take a gradient step
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return metrics

    def evaluate(self, input, targ, tag):
        metrics = {}

        if self.stochasticity == 'deterministic':
            mean = self.predict(input)
            mse = (mean - targ) ** 2
            xentropy = mse

        else:
            mean, logvar = self.predict(input)
            inv_var = torch.exp(-logvar)
            mse = (mean - targ) ** 2
            xentropy = mse * inv_var + logvar

            metrics['logvar/mean_train'] = logvar.mean()
            metrics['logvar/min_train'] = logvar.min()
            metrics['logvar/max_train'] = logvar.max()
            metrics['logvar/std_train'] = logvar.std()

        mse = mse.cpu().detach()
        xentropy = xentropy.cpu().detach()
        metrics[f'mse/mean_{tag}'] = mse.mean()
        metrics[f'mse/min_{tag}'] = mse.min()
        metrics[f'mse/max_{tag}'] = mse.max()
        metrics[f'mse/std_{tag}'] = mse.std()
        metrics[f'xentropy/mean_{tag}'] = xentropy.mean()

        return metrics