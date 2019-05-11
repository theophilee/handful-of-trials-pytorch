class ActionRepeat(object):
    def __init__(self, env, amount):
        self._env = env
        self.amount = amount
        self.task_hor = self._env._max_episode_steps
        self.num_steps = self._env._max_episode_steps // amount
        self.timestep = self._env.dt * self.amount

    @ property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        total_reward = 0

        for _ in range(self.amount):
            obs, reward, _, _ = self._env.step(action)
            total_reward += reward

        return obs, total_reward, False, {}

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)