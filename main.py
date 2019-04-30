import argparse

# To register environments
import env

from controller import MPC
from policy import Policy
from experiment import Experiment
from config import get_config

from utils import *


ALLOWED_ENVS = ["cartpole", "halfcheetah", "reacher3D", "pusher"]


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
    exp = Experiment(mpc, policy, args.env, args.logdir, args.savedir, cfg.exp_cfg)

    #exp.run_mpc_baseline()
    #exp.run_mpc_true_dynamics()
    #exp.run_pretrained_policy()
    #exp.run_behavior_cloning_basic()
    exp.run_dagger_basic()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah',
                        help='Env name: one of {}'.format(ALLOWED_ENVS))
    parser.add_argument('--logdir', type=str, default='runs/model-based',
                        help='Log directory for Tensorboard')
    parser.add_argument('--savedir', type=str, default='save/model-based',
                        help='Save directory')
    args = parser.parse_args()

    main(args)