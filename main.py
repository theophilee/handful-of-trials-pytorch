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
    cfg.mpc_cfg.model_cfg.hid_features = args.hid_features
    cfg.mpc_cfg.opt_cfg.iterations = args.iterations
    cfg.exp_cfg.train_freq = args.train_freq
    param_str = f'{args.ensemble_size}_{args.hid_features}_{args.iterations}_{args.train_freq}'

    # Model predictive control policy
    mpc = MPC(cfg.mpc_cfg)

    # Parameterized reactive policy
    policy = Policy(**cfg.policy_cfg)

    # Run experiment
    exp = Experiment(mpc, policy, args.env, param_str, args.logdir, args.savedir, cfg.exp_cfg)
    #exp.run_behavior_cloning_debug()
    #exp.run_train_model_debug()
    #exp.run_experiment_debug()
    exp.run_experiment_debug2()
    #exp.run_mpc_baseline()
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
    parser.add_argument('--hid_features', default=[200, 200, 200, 200],
                        type=lambda l: [int(x) for x in l.split(',')],
                        help='Hidden layers of dynamics model.')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations to perform during CEM optimization.')
    parser.add_argument('--train_freq', type=int, default=1,
                        help='Number of episodes to wait for before retraining model.')
    parser.add_argument('--expert-demos', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='If True, add expert demonstrations to dynamics model training set.')
    args = parser.parse_args()

    main(args)