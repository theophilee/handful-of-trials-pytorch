import gym
import numpy as np
import torch
from dotmap import DotMap

from .action_repeat import ActionRepeat


class Config:
    def __init__(self):
        env = gym.make("MyPusher-v2")
        action_repeat = 1
        self.env = ActionRepeat(env, action_repeat)

        self.obs_features = self.env.observation_space.shape[0]
        self.obs_features_preprocessed = self.obs_features
        self.act_features = self.env.action_space.shape[0]

    def obs_preproc(self, obs):
        return obs

    def pred_postproc(self, obs, pred):
        # Last three dimensions are the static goal
        if isinstance(obs, np.ndarray):
            return np.concatenate([pred + obs[:, :-3], obs[:, -3:]], axis=1)
        elif isinstance(obs, torch.Tensor):
            return torch.cat([pred + obs[:, :-3], obs[:, -3:]], dim=1)

    def targ_proc(self, obs, next_obs):
        # Last three dimensions are the static goal
        return next_obs[:, :-3] - obs[:, :-3]

    def get_reward(self, obs, act, next_obs):
        tip_pos, obj_pos, goal_pos = obs[:, 14:17], obs[:, 17:20], obs[:, 20:23]
        reward_near = -torch.norm(obj_pos - tip_pos, dim=1)
        reward_dist = -torch.norm(obj_pos - goal_pos, dim=1)
        reward_ctrl = -(act ** 2).sum(dim=1)
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        return reward

    def get_config(self):
        exp_cfg = DotMap({"env": self.env,
                          "expert_demos": False,
                          "init_steps": 100,
                          "total_steps": 10000,
                          "train_freq": 100,
                          "imaginary_steps": 500})

        model_cfg = DotMap({"ensemble_size": 5,
                            "in_features": self.obs_features_preprocessed + self.act_features,
                            "out_features": self.obs_features - 3,
                            "hid_features": [200, 200, 200, 200],
                            "activation": "swish",
                            "lr": 1e-3,
                            "weight_decay": 1e-4})

        opt_cfg = DotMap({"iterations": 5,
                          "popsize": 1000,
                          "num_elites": 50})

        mpc_cfg = DotMap({"env": self.env,
                          "plan_hor": 25,
                          "num_part": 20,
                          "batches_per_epoch": 100,
                          "obs_preproc": self.obs_preproc,
                          "pred_postproc": self.pred_postproc,
                          "targ_proc": self.targ_proc,
                          "get_reward": self.get_reward,
                          "model_cfg": model_cfg,
                          "opt_cfg": opt_cfg})

        policy_cfg = DotMap({"env": self.env,
                             "obs_features": self.obs_features_preprocessed,
                             "hid_features": [400, 300],
                             "activation": "tanh",
                             "batch_size": 32,
                             "lr": 1e-3,
                             "weight_decay": 0.,
                             "obs_preproc": self.obs_preproc})

        cfg = DotMap({"exp_cfg": exp_cfg,
                      "mpc_cfg": mpc_cfg,
                      "policy_cfg": policy_cfg})

        return cfg