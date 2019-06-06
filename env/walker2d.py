import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/walker2d.xml' % dir_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, action):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]

        reward_run = (posafter - posbefore) / self.dt
        reward_act = -1e-3 * np.square(action).sum()
        reward_alive = 1.0
        reward = reward_run + reward_act + reward_alive

        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Added first dimension of obs to be able to compute the reward from obs
        return np.concatenate([self.sim.data.qpos.flat,
                               np.clip(self.sim.data.qvel.flat, -10, 10)])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20