import os
import shutil
import random
import torch
import numpy as np
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


def create_directories(directories):
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)


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
    def __init__(self, ckpt_file='ckpt.pt', patience=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.ckpt_file = ckpt_file

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_ckpt(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_ckpt(val_loss, model)
            self.counter = 0

    def save_ckpt(self, val_loss, model):
        # Saves model when validation loss decreases
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}), saving model.')
        torch.save(model.state_dict(), self.ckpt_file)
        self.val_loss_min = val_loss

    def load_best(self, model):
        model.load_state_dict(torch.load(self.ckpt_file))