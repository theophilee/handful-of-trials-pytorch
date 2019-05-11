import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/swimmer.xml' % dir_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        reward_act = -1e-4 * np.square(action).sum()
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward = reward_fwd + reward_act

        done = False
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Added first dimension of obs to be able to compute the reward from obs
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel.flat])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()