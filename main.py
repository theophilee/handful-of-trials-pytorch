import argparse

import env # To register environments
from controller import MPC
from policy import Policy
from experiment import Experiment
from config import get_config
from config.action_repeat import ActionRepeat

from utils import *


ALLOWED_ENVS = ["ant", "cartpole", "half_cheetah", "humanoid", "inverted_pendulum", "pusher", "swimmer", "walker2d"]


def main(args):
    assert args.env in ALLOWED_ENVS

    set_random_seeds(args.seed)

    # Get configuration for environment
    cfg = get_config(args.env)

    # Overwrite configuration with command line arguments
    #cfg.exp_cfg.env = ActionRepeat(cfg.exp_cfg.env._env, args.action_repeat)
    cfg.mpc_cfg.model_cfg.stochasticity = args.stochasticity
    cfg.mpc_cfg.model_cfg.ensemble_size = args.ensemble_size
    cfg.mpc_cfg.model_cfg.hid_features = args.hid_features
    cfg.mpc_cfg.model_cfg.activation = args.activation
    cfg.mpc_cfg.model_cfg.weight_decay = args.weight_decay
    cfg.mpc_cfg.model_cfg.lr = args.lr
    cfg.mpc_cfg.opt_cfg.iterations = args.iterations
    cfg.mpc_cfg.plan_hor = args.plan_hor
    cfg.mpc_cfg.num_part = args.num_part
    param_str = (f'{args.stochasticity}_nets={args.ensemble_size}_hid={args.hid_features}'
                 f'_act={args.activation}_decay={args.weight_decay}_lr={args.lr}_seed={args.seed}'
                 f'_iter={args.iterations}_hor={args.plan_hor}_part={args.num_part}_expert={args.expert_demos}')

    # Model predictive control policy
    mpc = MPC(cfg.mpc_cfg)

    # Parameterized reactive policy
    policy = Policy(**cfg.policy_cfg)

    # Run experiment
    exp = Experiment(mpc, policy, args.env, param_str, args.logdir, args.savedir, cfg.exp_cfg)
    exp.run_experiment(args.load_controller, args.expert_demos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='half_cheetah',
                        help='Env name: one of {}.'.format(ALLOWED_ENVS))
    parser.add_argument('--logdir', type=str, default='runs/main',
                        help='Log directory for Tensorboard.')
    parser.add_argument('--savedir', type=str, default='save/main',
                        help='Save directory.')
    parser.add_argument('--seed', type=int, default=2,
                        help='Random seed.')
    parser.add_argument('--stochasticity', type=str, default='gaussian_bias',
                        help='One of "deterministic", "gaussian", "gaussian_bias"')
    parser.add_argument('--expert_demos', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='If True, add expert demonstrations to dynamics model training set.')
    parser.add_argument('--load_controller', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='If True, load mpc controller from previous experiment.')
    args = parser.parse_args()

    main(args)