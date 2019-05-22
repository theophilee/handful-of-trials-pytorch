import argparse

import env # To register environments
from controller import MPC
from policy import Policy
from experiment import Experiment
from config import get_config

from utils import *


ALLOWED_ENVS = ["cartpole", "half_cheetah", "swimmer", "pusher"]


def main(args):
    assert args.env in ALLOWED_ENVS

    set_random_seeds(args.seed)

    # Get configuration for environment
    cfg = get_config(args.env)

    # Overwrite configuration with command line arguments
    cfg.mpc_cfg.model_cfg.ensemble_size = args.ensemble_size
    cfg.mpc_cfg.model_cfg.activation = args.activation
    cfg.mpc_cfg.batches_per_epoch = args.batches_per_epoch
    cfg.exp_cfg.train_freq = args.train_freq
    param_str = f'{args.ensemble_size}_{args.activation}_{args.batches_per_epoch}_{args.train_freq}'

    # Model predictive control policy
    mpc = MPC(cfg.mpc_cfg)

    # Parameterized reactive policy
    policy = Policy(**cfg.policy_cfg)

    # Run experiment
    exp = Experiment(mpc, policy, args.env, param_str, args.logdir, args.savedir, cfg.exp_cfg)
    #exp.run_behavior_cloning_debug()
    #exp.run_train_model_debug()
    #exp.run_experiment_debug()
    exp.run_mpc_baseline()
    #exp.run_experiment()


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
    parser.add_argument('--activation', type=str, default='tanh',
                        help='Activation function for dynamics model.')
    parser.add_argument('--batches_per_epoch', type=int, default=100,
                        help='Number of batches per dynamics model training epoch.')
    parser.add_argument('--train_freq', type=int, default=1,
                        help='Number of episodes to wait for before retraining model.')
    parser.add_argument('--expert-demos', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='If True, add expert demonstrations to dynamics model training set.')
    args = parser.parse_args()

    main(args)