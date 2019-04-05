from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time

import numpy as np
from dotmap import DotMap


class Agent:
    """An general class for RL agents.
    """

    def __init__(self, params):
        """Initializes an agent.

        Arguments:
            params: (DotMap) A DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the
                    action noise if params.noisy_actions is True.
        """
        assert params.get("noisy_actions", False) is False
        self.env = params.env

        if isinstance(self.env, DotMap):
            raise ValueError("Environment must be provided to the agent at initialization.")

    def sample(self, horizon, policy):
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """

        times, rewards = [], []
        O, A, reward_sum, done = [self.env.reset()], [], 0, False

        policy.reset()
        for t in range(horizon):
            start = time.time()
            A.append(policy.act(O[t], t))
            times.append(time.time() - start)

            obs, reward, done, info = self.env.step(A[t])

            O.append(obs)
            reward_sum += reward
            rewards.append(reward)
            if done:
                break

        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }
