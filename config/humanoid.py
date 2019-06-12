import gym
from dotmap import DotMap
import torch

from .action_repeat import ActionRepeat


class Config:
    def __init__(self):
        env = gym.make("MyHumanoid-v2")
        action_repeat = 2
        self.env = ActionRepeat(env, action_repeat)

        self.obs_features = self.env.observation_space.shape[0]
        self.obs_features_preprocessed = self.obs_features - 14 * 4
        self.act_features = self.env.action_space.shape[0]

    def obs_preproc(self, obs):
        return obs[:, 14 * 4:]

    def pred_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def _mass_center(self, ob):
        mass, xpos = ob[:, :14].view(-1, 14, 1), ob[:, 14:14 * 4].view(-1, 14, 3)
        return ((mass * xpos).sum(dim=1) / mass.sum(dim=(1, 2)).view(-1, 1))[:, 0]

    def get_reward(self, obs, act, next_obs):
        reward_run = 1.25 * (self._mass_center(next_obs) - self._mass_center(obs)) / self.env.dt
        reward_act = -0.1 * (act ** 2).sum(dim=1) * self.env.amount
        reward_contact = max(-0.5e-6 * (next_obs[:, -14 * 6:] ** 2).sum(), -10) * self.env.amount
        reward_alive = 5.0 * self.env.amount
        reward = reward_run + reward_act + reward_contact + reward_alive
        done = torch.max(next_obs[:, 14 * 4] < 1.0, next_obs[:, 14 * 4] > 2.0).float()
        return reward, done

    def get_config(self):
        exp_cfg = DotMap({"env": self.env,
                          "expert_demos": False,
                          "init_steps": 200000,
                          "total_steps": 1000000,
                          "train_freq": 3000,
                          "imaginary_steps": 5000})

        model_cfg = DotMap({"deterministic": True,
                            "ensemble_size": 5,
                            "in_features": self.obs_features_preprocessed + self.act_features,
                            "out_features": self.obs_features,
                            "hid_features": [200, 200, 200, 200],
                            "activation": "swish",
                            "lr": 1e-3,
                            "weight_decay": 0})

        opt_cfg = DotMap({"iterations": 5,
                          "popsize": 1000,
                          "num_elites": 50})

        mpc_cfg = DotMap({"env": self.env,
                          "plan_hor": 20,
                          "num_part": 20,
                          "batches_per_epoch": 200,
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
                             "batch_size": 250,
                             "lr": 1e-3,
                             "weight_decay": 0.,
                             "obs_preproc":self.obs_preproc})

        cfg = DotMap({"exp_cfg": exp_cfg,
                      "mpc_cfg": mpc_cfg,
                      "policy_cfg": policy_cfg})

        return cfg
