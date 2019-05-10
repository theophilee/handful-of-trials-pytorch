class ActionRepeat(object):
    def __init__(self, env, amount):
        self._env = env
        self._amount = amount
        self._env._task_hor = self._env._max_episode_steps
        self._env._max_episode_steps = self._env._max_episode_steps // amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        total_reward = 0

        for _ in range(self._amount):
            obs, reward, _, _ = self._env.step(action)
            total_reward += reward

        return obs, total_reward, False, {}

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)