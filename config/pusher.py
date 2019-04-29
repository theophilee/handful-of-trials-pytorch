import gym
import torch
from dotmap import DotMap


class Config:
    def __init__(self):
        self.env = gym.make("MyPusher-v0")
        self.task_hor = 150
        self.num_rollouts = 100
        self.in_features, self.out_features = 27, 20

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.prev_ac_goal_pos = None
        self.goal_pos_gpu = None

    def obs_preproc(self, obs):
        return obs

    def pred_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def get_cost(self, obs, act, next_obs):
        assert isinstance(next_obs, torch.Tensor)

        cost_act = 0.1 * (act ** 2).sum(dim=1)

        to_w, og_w = 0.5, 1.25
        tip_pos, obj_pos, goal_pos = next_obs[:, 14:17], next_obs[:, 17:20], self.env.ac_goal_pos

        should_replace = False
        if self.prev_ac_goal_pos is not None and (self.prev_ac_goal_pos == goal_pos).all() is False:
            should_replace = True
        elif self.goal_pos_gpu is None:
            should_replace = True

        if should_replace:
            self.goal_pos_gpu = torch.from_numpy(goal_pos).float().to(self.device)
            self.prev_ac_goal_pos = goal_pos

        tip_obj_dist = (tip_pos - obj_pos).abs().sum(dim=1)
        obj_goal_dist = (self.goal_pos_gpu - obj_pos).abs().sum(dim=1)
        cost_obs = to_w * tip_obj_dist + og_w * obj_goal_dist

        return cost_act + cost_obs

    def get_config(self):
        exp_cfg = DotMap({"env": self.env,
                          "task_hor": self.task_hor,
                          "num_rollouts": self.num_rollouts,
                          "num_imagined_rollouts": 2})

        model_cfg = DotMap({"ensemble_size": 1,
                            "in_features": self.in_features,
                            "out_features": self.out_features,
                            "hid_features": [200, 200],
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
                          "get_cost": self.get_cost,
                          "reset_fns": [],
                          "model_cfg": model_cfg,
                          "opt_cfg": opt_cfg})

        policy_cfg = DotMap({"env": self.env,
                             "hid_features": [200],
                             "activation": "relu",
                             "train_epochs": 20,
                             "batch_size": 32,
                             "lr": 1e-3,
                             "weight_decay": 0.})

        cfg = DotMap({"exp_cfg": exp_cfg,
                      "mpc_cfg": mpc_cfg,
                      "policy_cfg": policy_cfg})

        return cfg