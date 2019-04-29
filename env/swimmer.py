import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class SwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/swimmer.xml' % dir_path, 4)
        utils.EzPickle.__init__(self)

    def _step(self, act):
        self.prev_qpos = np.copy(self.model.data.qpos.flat)
        self.do_simulation(act, self.frame_skip)
        ob = self._get_obs()

        reward_act = -0.0001 * np.square(act).sum()
        reward_fwd = ob[0]
        reward = reward_fwd + reward_act

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        # Added first dimension of obs to be able to compute the reward from obs
        return np.concatenate([
            (self.model.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.model.data.qpos.flat)
        return self._get_obs()