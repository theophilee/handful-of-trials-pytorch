import argparse

import env # To register environments
from controller import MPC
from policy import Policy
from experiment import Experiment
from config import get_config

from utils import *


ALLOWED_ENVS = ["cartpole", "half_cheetah", "hopper", "swimmer"]


def main(args):
    assert args.env in ALLOWED_ENVS

    set_random_seeds(args.seed)

    # Get configuration for environment
    cfg = get_config(args.env)

    # Overwrite configuration with command line arguments
    param_str = ""

    # Model predictive control policy
    mpc = MPC(param_str, cfg.mpc_cfg)

    # Parameterized reactive policy
    policy = Policy(param_str, **cfg.policy_cfg)

    # Run experiment
    exp = Experiment(mpc, policy, args.env, param_str, args.logdir, args.savedir, cfg.exp_cfg)
    exp.run_mpc_baseline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='cartpole',
                        help='Env name: one of {}.'.format(ALLOWED_ENVS))
    parser.add_argument('--logdir', type=str, default='runs/main',
                        help='Log directory for Tensorboard.')
    parser.add_argument('--savedir', type=str, default='save/main',
                        help='Save directory.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    args = parser.parse_args()

    main(args)