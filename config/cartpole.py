import gym
import numpy as np
import torch
from dotmap import DotMap

from .action_repeat import ActionRepeat


class Config:
    def __init__(self):
        env = gym.make("MyCartpole-v0")
        action_repeat = 4
        self.env = ActionRepeat(env, action_repeat)

        self.obs_features = self.env.observation_space.shape[0]
        self.obs_features_preprocessed = self.obs_features + 1
        self.act_features = self.env.action_space.shape[0]

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.ee_sub = torch.tensor([0.0, 0.6], device=self.device, dtype=torch.float)

    def obs_preproc(self, obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[:, :1], np.sin(obs[:, 1:2]), np.cos(obs[:, 1:2]), obs[:, 2:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([obs[:, :1], obs[:, 1:2].sin(), obs[:, 1:2].cos(), obs[:, 2:]], dim=1)

    def pred_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def get_cost(self, obs, act, next_obs):
        ee_pos = self.get_ee_pos(next_obs)
        ee_pos -= self.ee_sub
        ee_pos = ee_pos ** 2
        ee_pos = -ee_pos.sum(dim=1)
        cost_obs = -(ee_pos / (0.6 ** 2)).exp()
        cost_act = 0.01 * (act ** 2).sum(dim=1)
        return cost_obs + cost_act

    def get_ee_pos(self, obs):
        x0, theta = obs[:, :1], obs[:, 1:2]
        return torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)

    def get_config(self):
        exp_cfg = DotMap({"env": self.env,
                          "expert_demos": False,
                          "num_init_rollouts": 2,
                          "num_rollouts": 15,
                          "train_freq": 1,
                          "num_imagined_rollouts": 2})

        model_cfg = DotMap({"ensemble_size": 1,
                            "in_features": self.obs_features_preprocessed + self.act_features,
                            "out_features": self.obs_features,
                            "hid_features": [200, 200, 200, 200],
                            "activation": "relu",
                            "lr": 1e-3,
                            "weight_decay": 1e-4})

        opt_cfg = DotMap({"iterations": 10,
                          "popsize": 1000,
                          "num_elites": 50})

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
                             "obs_features": self.obs_features_preprocessed,
                             "hid_features": [400, 300],
                             "activation": "relu",
                             "batch_size": 250,
                             "lr": 1e-3,
                             "weight_decay": 0.,
                             "obs_preproc": self.obs_preproc})

        cfg = DotMap({"exp_cfg": exp_cfg,
                      "mpc_cfg": mpc_cfg,
                      "policy_cfg": policy_cfg})

        return cfg