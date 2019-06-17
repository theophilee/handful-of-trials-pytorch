from .layers import *


ACTIVATIONS = {'relu': nn.ReLU(), 'swish': Swish(), 'tanh': nn.Tanh()}
TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def log_statistics(dict, tensor, key):
    dict[key + '_mean'] = tensor.mean()
    dict[key + '_min'] = tensor.min()
    dict[key + '_max'] = tensor.max()
    dict[key + '_std'] = tensor.std()
    dict[key + '_median'] = tensor.median()


class BootstrapEnsemble:
    def __init__(self, stochasticity, ensemble_size, in_features, out_features, hid_features,
                 activation, lr, weight_decay, dropout):
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
            dropout (float): dropout probability
        """
        self.stochasticity = stochasticity
        self.net = self._make_network(ensemble_size, in_features, out_features, hid_features,
                                      activation, dropout).to(TORCH_DEVICE)

        special, default = [], []
        for name, param in self.net.named_parameters():
            if name.split('.')[-1] in ['logvar', 'min_logvar', 'max_logvar']:
                special.append(param)
            else:
                default.append(param)
        params = [{'params': special, 'weight_decay': 0}, {'params': default}]
        self.optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def _make_network(self, ensemble_size, in_features, out_features, hid_features, activation,
                      dropout):
        layers = []
        for in_f, out_f in zip([in_features] + hid_features, hid_features):
            layers.append(BootstrapLinear(ensemble_size, in_f, out_f))
            layers.append(ACTIVATIONS[activation])
            layers.append(nn.Dropout(p=dropout))

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
            return mean + torch.randn_like(mean, device=input.device) * logvar.exp().sqrt()

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
            rescaled_mse = mse * inv_var
            xentropy = rescaled_mse + logvar
            loss = xentropy.mean()

            if self.stochasticity == 'gaussian':
                # Small special regularization for max and min log variance parameters
                reg = 1e-6 * (self.net[-1].max_logvar.mean() - self.net[-1].min_logvar.mean())
                loss += reg

            log_statistics(metrics, logvar, 'logvar/train')
            log_statistics(metrics, rescaled_mse, 'rescaled_mse/train')

        log_statistics(metrics, mse, 'mse/train')
        log_statistics(metrics, xentropy, 'xentropy/train')

        # Take a gradient step
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return metrics

    @torch.no_grad()
    def evaluate_val(self, input, targ):
        metrics = {}

        if self.stochasticity == 'deterministic':
            mean = self.predict(input)
            mse = (mean - targ) ** 2
            xentropy = mse

        else:
            mean, logvar = self.predict(input)
            inv_var = torch.exp(-logvar)
            mse = (mean - targ) ** 2
            rescaled_mse = mse * inv_var
            xentropy = rescaled_mse + logvar

            log_statistics(metrics, logvar, 'logvar/val')
            log_statistics(metrics, rescaled_mse, 'rescaled_mse/val')

        log_statistics(metrics, mse, 'mse/val')
        log_statistics(metrics, xentropy, 'xentropy/val')

        return metrics

    @torch.no_grad()
    def evaluate_test(self, input, targ):
        metrics = {}

        if self.stochasticity == 'deterministic':
            mean = self.predict(input)

        else:
            mean, logvar = self.predict(input)
            sample = mean + torch.randn_like(mean, device=input.device) * logvar.exp().sqrt()

            log_statistics(metrics, (sample - targ) ** 2, 'test/mse_sample_individual')
            log_statistics(metrics, (sample.mean(dim=0) - targ) ** 2, 'test/mse_sample_ensemble')

        log_statistics(metrics, (mean - targ) ** 2, 'test/mse_mean_individual')
        log_statistics(metrics, (mean.mean(dim=0) - targ) ** 2, 'test/mse_mean_ensemble')

        return metrics