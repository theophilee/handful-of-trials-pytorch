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
    cfg.mpc_cfg.model_cfg.ensemble_size = args.ensemble_size
    param_str = '%d' % args.ensemble_size

    # Model predictive control policy
    mpc = MPC(cfg.mpc_cfg)

    # Parameterized reactive policy
    policy = Policy(**cfg.policy_cfg)

    # Run experiment
    exp = Experiment(mpc, policy, args.env, param_str, args.logdir, args.savedir, cfg.exp_cfg)
    #exp.run_behavior_cloning_debug()
    #exp.run_train_model_debug()
    exp.run_mpc_baseline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='half_cheetah',
                        help='Env name: one of {}.'.format(ALLOWED_ENVS))
    parser.add_argument('--logdir', type=str, default='runs/main',
                        help='Log directory for Tensorboard.')
    parser.add_argument('--savedir', type=str, default='save/main',
                        help='Save directory.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--ensemble-size', type=int, default=5,
                        help='Number of bootstrap ensemble dynamics models.')
    args = parser.parse_args()

    main(args)