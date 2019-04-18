import gym
import numpy as np
import torch
from dotmap import DotMap


class Config:
    def __init__(self):
        self.env = gym.make("MBRLCartpole-v0")
        self.task_hor = 200
        self.num_rollouts = 15
        self.in_features, self.out_features = 6, 4

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.ee_sub = torch.tensor([0.0, 0.6], device=self.device, dtype=torch.float)

    def obs_preproc(self, obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate(
                [np.sin(obs[:, 1:2]),  np.cos(obs[:, 1:2]), obs[:, :1], obs[:, 2:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat(
                [obs[:, 1:2].sin(), obs[:, 1:2].cos(), obs[:, :1], obs[:, 2:]], dim=1)

    def pred_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def get_cost_obs(self, obs):
        ee_pos = self.get_ee_pos(obs)
        ee_pos -= self.ee_sub
        ee_pos = ee_pos ** 2
        ee_pos = -ee_pos.sum(dim=1)
        return -(ee_pos / (0.6 ** 2)).exp()

    def get_cost_acts(self, acts):
        return 0.01 * (acts ** 2).sum(dim=1)

    def get_ee_pos(self, obs):
        x0, theta = obs[:, :1], obs[:, 1:2]
        return torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)

    def get_config(self):
        exp_cfg = DotMap({"env": self.env,
                          "task_hor": self.task_hor,
                          "num_rollouts": self.num_rollouts})
        
        """
        model_cfg = DotMap({"ensemble_size": 5,
                            "in_features": self.in_features,
                            "out_features": self.out_features,
                            "hid_features": [500, 500, 500],
                            "activation": "swish",
                            "lr": 1e-3,
                            "weight_decay": 1e-4})
        """

        model_cfg = DotMap({"ensemble_size": 1,
                            "in_features": self.in_features,
                            "out_features": self.out_features,
                            "hid_features": [200],
                            "activation": "relu",
                            "lr": 1e-3,
                            "weight_decay": 1e-4})

        opt_cfg = DotMap({"max_iters": 5,
                          "popsize": 500,
                          "num_elites": 50,
                          "epsilon": 0.01,
                          "alpha": 0.1})

        mpc_cfg = DotMap({"env": self.env,
                          "plan_hor": 25,
                          "num_part": 20,
                          "train_epochs": 5,
                          "batch_size": 32,
                          "obs_preproc": self.obs_preproc,
                          "pred_postproc": self.pred_postproc,
                          "targ_proc": self.targ_proc,
                          "get_cost_obs": self.get_cost_obs,
                          "get_cost_acts": self.get_cost_acts,
                          "reset_fns": [],
                          "model_cfg": model_cfg,
                          "opt_cfg": opt_cfg})

        cfg = DotMap({"exp_cfg": exp_cfg,
                      "mpc_cfg": mpc_cfg})

        return cfg
