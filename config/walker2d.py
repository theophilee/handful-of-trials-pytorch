import gym
from dotmap import DotMap

from .action_repeat import ActionRepeat


class Config:
    def __init__(self):
        env = gym.make("MyWalker2d-v2")
        action_repeat = 2
        self.env = ActionRepeat(env, action_repeat)

        self.obs_features = self.env.observation_space.shape[0]
        self.obs_features_preprocessed = self.obs_features - 1
        self.act_features = self.env.action_space.shape[0]

    def obs_preproc(self, obs):
        return obs[:, 1:]

    def pred_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def get_reward(self, obs, act, next_obs):
        reward_run = (next_obs[:, 0] - obs[:, 0]) / self.env.dt
        reward_act = -1e-3 * (act ** 2).sum(dim=1) * self.env.amount
        reward_alive = 1.0 * self.env.amount
        reward = reward_run + reward_act + reward_alive
        height, ang = next_obs[:, 1], next_obs[:, 2]
        done = (1 - (height > 0.8) * (height < 2.0) * (ang > -1.0) * (ang < 1.0)).float()
        return reward, done

    def get_config(self):
        exp_cfg = DotMap({"env": self.env,
                          "expert_demos": False,
                          "init_steps": 5000,
                          "total_steps": 100000,
                          "train_freq": 1000,
                          "imaginary_steps": 5000})

        model_cfg = DotMap({"ensemble_size": 5,
                            "in_features": self.obs_features_preprocessed + self.act_features,
                            "out_features": self.obs_features,
                            "hid_features": [200, 200, 200, 200],
                            "activation": "swish",
                            "lr": 1e-3,
                            "weight_decay": 1e-4,
                            "dropout": 0})

        opt_cfg = DotMap({"iterations": 5,
                          "popsize": 1000,
                          "num_elites": 50})

        mpc_cfg = DotMap({"env": self.env,
                          "plan_hor": 25,
                          "num_part": 20,
                          "train_epochs": 10,
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
                             "batch_size": 250,
                             "lr": 1e-3,
                             "weight_decay": 0.,
                             "obs_preproc":self.obs_preproc})

        cfg = DotMap({"exp_cfg": exp_cfg,
                      "mpc_cfg": mpc_cfg,
                      "policy_cfg": policy_cfg})

        return cfg
