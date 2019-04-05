from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import os
import pprint

from dotmap import DotMap

from experiment import MBExperiment
from controller import MPC
from config import create_config
import env # We run this so that the env is registered

import torch
import numpy as np
import random
import tensorflow as tf

ALLOWED_ENVS = ["cartpole", "reacher", "pusher", "halfcheetah"]


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


def main(env, ctrl_args, overrides, logdir):
    set_random_seeds(0)

    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, "MPC", ctrl_args, overrides, logdir)
    cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)
    exp = MBExperiment(cfg.exp_cfg)

    os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    exp.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, default='pusher',
                        help='Env name: one of {}'.format(ALLOWED_ENVS))
    parser.add_argument('-ca', '--ctrl_args', action='append', nargs=2, default=[],
                        help='Controller arguments, see tf implementation')
    parser.add_argument('-o', '--overrides', action='append', nargs=2, default=[],
                        help='Override default parameters, see tf implementation')
    parser.add_argument('-logdir', type=str, default='log',
                        help='Log directory')
    args = parser.parse_args()

    main(args.env, args.ctrl_args, args.overrides, args.logdir)