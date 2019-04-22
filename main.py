import os
import argparse
import torch
import numpy as np
import random
import tensorflow as tf

# To register environments
import env

from controller import MPC
from policy import Policy
from experiment import Experiment
from config import get_config


ALLOWED_ENVS = ["cartpole", "reacher", "pusher", "halfcheetah"]


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


def create_directories(directories):
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)


def main(args):
    assert args.env in ALLOWED_ENVS

    # Set random seeds
    set_random_seeds(0)

    # Create log and save directories
    create_directories([args.logdir, args.savedir])

    # Get configuration for environment
    cfg = get_config(args.env)

    # Model predictive control policy
    mpc = MPC(cfg.mpc_cfg)

    # Parameterized reactive policy
    policy = Policy(**cfg.policy_cfg)

    # Run experiment
    exp = Experiment(mpc, policy, args.logdir, args.savedir, cfg.exp_cfg)
    #exp.run_baseline()
    exp.run_behavior_cloning()
    #exp.run_DAgger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole',
                        help='Env name: one of {}'.format(ALLOWED_ENVS))
    parser.add_argument('--logdir', type=str, default='logs/cartpole',
                        help='Log directory for Tensorboard')
    parser.add_argument('--savedir', type=str, default='save/cartpole',
                        help='Save directory for models')
    args = parser.parse_args()

    main(args)
