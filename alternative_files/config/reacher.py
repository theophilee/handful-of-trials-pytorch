import gym
import numpy as np
import torch
from dotmap import DotMap


class Config:
    def __init__(self):
        self.env = gym.make("MBRLReacher3D-v0")
        self.task_hor = 150
        self.num_rollouts = 100
        self.in_features, self.out_features = 24, 17

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def obs_preproc(self, obs):
        return obs

    def pred_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def update_goal(self):
        self.goal = self.env.goal

    def get_cost_obs(self, obs):
        assert isinstance(obs, torch.Tensor)
        assert self.goal is not None

        obs = obs.detach().cpu().numpy()

        ee_pos = self.get_ee_pos(obs)
        dis = ee_pos - self.goal

        cost = np.sum(np.square(dis), axis=1)

        return torch.from_numpy(cost).float().to(self.device)

    def get_cost_acts(self, acts):
        return 0.01 * (acts ** 2).sum(dim=1)

    def get_ee_pos(self, states):
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = \
            states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:4], states[:, 4:5], states[:, 5:6], states[:, 6:]

        rot_axis = np.concatenate([np.cos(theta2) * np.cos(theta1), np.cos(theta2) * np.sin(theta1), -np.sin(theta2)],
                                  axis=1)

        rot_perp_axis = np.concatenate([-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1)
        cur_end = np.concatenate([
            0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
            0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
            -0.4 * np.sin(theta2)
        ], axis=1)

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30] = \
                rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
            new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
            rot_axis, rot_perp_axis, cur_end = new_rot_axis, new_rot_perp_axis, cur_end + length * new_rot_axis

        return cur_end

    def get_config(self):
        exp_cfg = DotMap({"env": self.env,
                          "task_hor": self.task_hor,
                          "num_rollouts": self.num_rollouts})

        model_cfg = DotMap({"ensemble_size": 5,
                            "in_features": self.in_features,
                            "out_features": self.out_features,
                            "hid_features": [200, 200],
                            "activation": "swish",
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
                          "reset_fns": [self.update_goal],
                          "model_cfg": model_cfg,
                          "opt_cfg": opt_cfg})

        cfg = DotMap({"exp_cfg": exp_cfg,
                      "mpc_cfg": mpc_cfg})

        return cfg