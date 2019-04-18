iport argparse

import torch
import numpy as np
import random
import tensorflow as tf

# To register environments
import env

from controller import MPC
from experiment import Experiment
from config import get_config


ALLOWED_ENVS = ["cartpole", "reacher", "pusher", "halfcheetah"]


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


def main(args):
    assert args.env in ALLOWED_ENVS

    # Set random seeds
    set_random_seeds(0)

    # Get configuration for environment
    cfg = get_config(args.env)

    # Model predictive control policy
    policy = MPC(cfg.mpc_cfg)

    # Run experiment
    exp = Experiment(policy, args.logdir, cfg.exp_cfg)
    exp.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole',
                        help='Env name: one of {}'.format(ALLOWED_ENVS))
    parser.add_argument('--logdir', type=str, default='logs/cartpole',
                        help='Log directory for Tensorboard')
    args = parser.parse_args()

    main(args)
