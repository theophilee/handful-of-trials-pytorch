import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


def mass_center(ob):
    mass, xpos = ob[:14].reshape(14, 1), ob[14:14 * 4].reshape(14, 3)
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/humanoid.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        pos_before = mass_center(self._get_obs())
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        pos_after = mass_center(ob)

        reward_run = 1.25 * (pos_after - pos_before) / self.dt
        reward_act = -0.1 * np.square(action).sum()
        reward_contact = max(-0.5e-6 * np.square(self.sim.data.cfrc_ext).sum(), -10)
        reward_alive = 5.0
        reward = reward_run + reward_act + reward_contact + reward_alive

        done = self.sim.data.qpos[2] < 1.0 or self.sim.data.qpos[2] > 2.0
        return ob, reward, done, {}

    def _get_obs(self):
        # Added body_mass and xipos to be able to compute the reward from obs
        return np.concatenate([self.model.body_mass.flat,        # (14,)
                               self.sim.data.xipos.flat,         # (14, 3)
                               self.sim.data.qpos.flat[2:],
                               self.sim.data.qvel.flat,
                               self.sim.data.cinert.flat,
                               self.sim.data.cvel.flat,
                               self.sim.data.qfrc_actuator.flat,
                               self.sim.data.cfrc_ext.flat])     # (14, 6)

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20