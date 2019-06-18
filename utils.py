import os
import shutil
import random
import torch
import numpy as np
import uuid
from tensorboardX import SummaryWriter


TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def numpy_to_device(arr):
    return torch.from_numpy(arr).float().to(TORCH_DEVICE)


def numpy_from_device(tensor):
    return tensor.cpu().detach().numpy()


def log_statistics(dict, tensor, key):
    dict[key + '_mean'] = tensor.mean()
    dict[key + '_median'] = tensor.median()


class Logger:
    """Logging with TensorboardX.
    """
    def __init__(self, logdir):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        try:
            shutil.rmtree(logdir)
        except FileNotFoundError:
            pass

        self.writer = SummaryWriter(logdir)

    def log_scalar(self, tag, value, step):
        """Log scalar value.
        """
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, array, step):
        """Log histogram of numpy array of values.
        """
        self.writer.add_histogram(tag, array, step)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=True):
        """
        Arguments:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.ckpt_file = str(uuid.uuid1()) + '.ckpt'

    def step(self, val_loss, network, info={}):
        """
        Arguments:
            val_loss (float): Validation loss.
            network (torch.nn.Module): Network to save if validation loss decreases.
            info (dict): Dictionary with extra information, to be loaded at the end.
        """
        if val_loss >= self.val_loss_min:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self._save_ckpt(val_loss, network, info)
            self.val_loss_min = val_loss
            self.counter = 0

    def _save_ckpt(self, val_loss, network, info):
        # Save network and additional info when validation loss decreases
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}), saving network.')
        torch.save(network.state_dict(), self.ckpt_file)
        self.info_best = info

    def load_best(self, network):
        # Load best network, clean-up checkpoint file and return additional info for best network
        if self.verbose:
            print(f'Loading best network with validation loss {self.val_loss_min:.6f}.')
        network.load_state_dict(torch.load(self.ckpt_file))
        os.remove(self.ckpt_file)
        return self.info_best


class Metrics:
    """Keep track of metrics over time in a dictionary.
    """
    def __init__(self):
        self.metrics = {}
        self.count = 0

    def store(self, new_metrics):
        self.count += 1
        for key in new_metrics:
            if key in self.metrics:
                self.metrics[key] += new_metrics[key]
            else:
                self.metrics[key] = new_metrics[key]

    def average(self):
        return {k: v / self.count for k, v in self.metrics.items()}