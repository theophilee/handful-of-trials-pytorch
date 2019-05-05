import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, act):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(act, self.frame_skip)
        ob = self._get_obs()

        reward_act = -0.1 * np.square(act).sum()
        reward_run = ob[0]
        reward = reward_run + reward_act

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        # Added first dimension of obs to be able to compute the reward from obs
        return np.concatenate([
            (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5