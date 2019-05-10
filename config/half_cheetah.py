import gym
import numpy as np
import torch
from dotmap import DotMap

from .action_repeat import ActionRepeat


class Config:
    def __init__(self):
        env = gym.make("MyHalfCheetah-v2")
        self.env = ActionRepeat(env, 4)

    def obs_preproc(self, obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate(
                [obs[:, 1:2], np.sin(obs[:, 2:3]), np.cos(obs[:, 2:3]), obs[:, 3:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat(
                [obs[:, 1:2], obs[:, 2:3].sin(), obs[:, 2:3].cos(), obs[:, 3:]], dim=1)

    def pred_postproc(self, obs, pred):
        if isinstance(obs, np.ndarray):
            return np.concatenate([pred[:, :1], obs[:, 1:] + pred[:, 1:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([pred[:, :1], obs[:, 1:] + pred[:, 1:]], dim=1)

    def targ_proc(self, obs, next_obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([next_obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([next_obs[:, :1], next_obs[:, 1:] - obs[:, 1:]], dim=1)

    def get_cost(self, obs, act, next_obs):
        cost_act = 0.1 * (act ** 2).sum(dim=1)
        cost_run = -next_obs[:, 0]
        return cost_act + cost_run

    def get_config(self):
        exp_cfg = DotMap({"env": self.env,
                          "expert_demos": False,
                          "num_init_rollouts": 10,
                          "num_rollouts": 300,
                          "train_freq": 3,
                          "num_imagined_rollouts": 2})

        model_cfg = DotMap({"ensemble_size": 1,
                            "in_features": 24,
                            "out_features": 18,
                            "hid_features": [200, 200, 200, 200],
                            "activation": "relu",
                            "lr": 1e-3,
                            "weight_decay": 1e-4})

        opt_cfg = DotMap({"iterations": 10,
                          "popsize": 1000,
                          "num_elites": 60})

        mpc_cfg = DotMap({"env": self.env,
                          "plan_hor": 12,
                          "num_part": 20,
                          "batch_size": 32,
                          "obs_preproc": self.obs_preproc,
                          "pred_postproc": self.pred_postproc,
                          "targ_proc": self.targ_proc,
                          "get_cost": self.get_cost,
                          "model_cfg": model_cfg,
                          "opt_cfg": opt_cfg})

        policy_cfg = DotMap({"env": self.env,
                             "obs_features": 18,
                             "hid_features": [400, 300],
                             "activation": "relu",
                             "batch_size": 250,
                             "lr": 1e-3,
                             "weight_decay": 0.,
                             "obs_preproc":self.obs_preproc})

        cfg = DotMap({"exp_cfg": exp_cfg,
                      "mpc_cfg": mpc_cfg,
                      "policy_cfg": policy_cfg})

        return cfg