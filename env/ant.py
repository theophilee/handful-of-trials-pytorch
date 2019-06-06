import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/ant.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        reward_run = (xposafter - xposbefore) / self.dt
        reward_act = -0.5 * np.square(action).sum()
        reward_contact = -0.5e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        reward_alive = 1.0
        reward = reward_run + reward_act + reward_contact + reward_alive

        done = self.sim.data.qpos[2] < 0.2 or self.sim.data.qpos[2] > 1.0
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Added first two dimensions of obs to be able to compute the reward from obs
        return np.concatenate([self.sim.data.qpos.flat,
                               self.sim.data.qvel.flat,
                               np.clip(self.sim.data.cfrc_ext, -1, 1).flat]) # (14, 6)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5