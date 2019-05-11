import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class CartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/cartpole.xml' % dir_path, 2)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        reward_obs = np.exp(-np.sum(np.square(self._get_ee_pos(obs) - np.array([0.0, 0.6]))) / (0.6 ** 2))
        reward_act = -0.01 * np.sum(np.square(action))
        reward = reward_obs + reward_act

        done = False
        return obs, reward, done, {}

    @staticmethod
    def _get_ee_pos(obs):
        x0, theta = obs[0], obs[1]
        return np.array([x0 - 0.6 * np.sin(theta), -0.6 * np.cos(theta)])

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent