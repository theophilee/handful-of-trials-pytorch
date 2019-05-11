import gym
from dotmap import DotMap

from .action_repeat import ActionRepeat


class Config:
    def __init__(self):
        env = gym.make("MySwimmer-v2")
        action_repeat = 9
        self.env = ActionRepeat(env, action_repeat)

        self.obs_features = self.env.observation_space.shape[0]
        self.obs_features_preprocessed = self.obs_features
        self.act_features = self.env.action_space.shape[0]

    def obs_preproc(self, obs):
        return obs

    def pred_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def get_cost(self, obs, act, next_obs):
        cost_act = 1e-4 * (act ** 2).sum(dim=1)
        cost_run = -(next_obs[:, 0] - obs[:, 0]) / self.env.timestep
        return cost_act + cost_run

    def get_config(self):
        exp_cfg = DotMap({"env": self.env,
                          "expert_demos": False,
                          "num_init_rollouts": 10,
                          "num_rollouts": 300,
                          "train_freq": 3,
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
                          "num_elites": 20})

        mpc_cfg = DotMap({"env": self.env,
                          "plan_hor": 16,
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
                             "hid_features": [200, 200],
                             "activation": "relu",
                             "batch_size": 250,
                             "lr": 1e-3,
                             "weight_decay": 0.,
                             "obs_preproc": self.obs_preproc})

        cfg = DotMap({"exp_cfg": exp_cfg,
                      "mpc_cfg": mpc_cfg,
                      "policy_cfg": policy_cfg})

        return cfg