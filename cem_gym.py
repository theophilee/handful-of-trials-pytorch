"""
MPC CEM planning on gym environments using the ground truth dynamics.
Example usage:
python cem_gym.py MyHalfCheetah-v2 -a gaussian -r 4 -l 12
"""
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial
import gym
import os

import env # Register environments
from utils import Logger


class ActionRepeat(object):
    def __init__(self, env, amount):
        self._env = env
        self.amount = amount
        self.num_steps = self._env._max_episode_steps // amount

    @ property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def sim(self):
        return self._env.sim

    def step(self, action):
        total_reward = 0

        for _ in range(self.amount):
            obs, reward, done, _ = self._env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, {}

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)


def initializer(env):
    global global_env
    global_env = env


def evaluate(actions, state):
    global_env.reset()
    global_env.sim.set_state(state)

    score = 0
    for action in actions:
        _, reward, done, _ = global_env.step(action)
        score += reward
        if done:
            break

    return score


def gaussian_cem(state, pool, action_space, horizon, proposals, topk, iterations):
    action_bound = action_space.high[0]
    mean = np.zeros((horizon,) + action_space.shape)
    std = np.ones((horizon,) + action_space.shape) * action_bound

    for _ in range(iterations):
        plans = np.random.normal(mean, std, size=(proposals,) + mean.shape)
        scores = pool.map(partial(evaluate, state=state), plans.clip(-action_bound, action_bound))
        elites = plans[np.argsort(scores)][-topk:]
        mean, std = elites.mean(axis=0), elites.std(axis=0)

    # Return first action of mean of last iteration
    return mean[0].clip(-action_bound, action_bound)


def nonparametric_cem(state, pool, action_space, horizon, proposals, topk, iterations, sigma):
    action_bound = action_space.high[0]
    plans = np.random.randn(proposals, horizon, *action_space.shape) * action_bound

    for _ in range(iterations - 1):
        scores = pool.map(partial(evaluate, state=state), plans.clip(-action_bound, action_bound))
        elites = plans[np.argsort(scores)][-topk:]
        means = elites[np.random.randint(topk, size=proposals)]
        noise = np.random.randn(*means.shape) * sigma * action_bound
        plans = means + noise

    # Return first action of best plan of last iteration
    scores = pool.map(partial(evaluate, state=state), plans.clip(-action_bound, action_bound))
    return plans[np.argsort(scores)][-1, 0].clip(-action_bound, action_bound)


def main(args):
    param_str = (f'{args.env}_{args.algo}_rep={args.repeat}_hor={args.horizon}_prop={args.proposals}'
                 f'_iter={args.iterations}_sigma={args.sigma}')

    env = gym.make(args.env)
    env = ActionRepeat(env, args.repeat)

    # Pool of workers, each has its own copy of global environment variable
    pool = Pool(32, initializer, [env])

    if args.algo == 'gaussian':
        planner = partial(gaussian_cem, pool=pool, action_space=env.action_space, horizon=args.horizon,
                          proposals=args.proposals, topk=args.topk, iterations=args.iterations)
    elif args.algo == 'nonparametric':
        planner = partial(nonparametric_cem, pool=pool, action_space=env.action_space, horizon=args.horizon,
                          proposals=args.proposals, topk=args.topk, iterations=args.iterations,
                          sigma = args.sigma)

    scores = np.zeros(args.episodes)
    observations = np.zeros((args.episodes, env.num_steps + 1) + env.observation_space.shape)
    actions = np.zeros((args.episodes, env.num_steps) + env.action_space.shape)

    for i in range(args.episodes):
        logger = Logger(os.path.join(args.logdir, f'{param_str}_run{i}'))
        observations[i, 0] = env.reset()

        for t in range(env.num_steps):
            state = env.sim.get_state()
            actions[i, t] = planner(state)
            observations[i, t + 1], reward, _, _ = env.step(actions[i, t])
            scores[i] += reward
            logger.log_scalar('reward', scores[i], t)

        print(scores[i])

    print(param_str)
    print('Mean score:         ', scores.mean())
    print('Standard deviation: ', scores.std())

    if args.save:
        path = os.path.join(args.savedir, args.env)
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, 'obs'), observations)
        np.save(os.path.join(path, 'act'), actions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env',
                        help='OpenAI gym environment to load.')
    parser.add_argument('-a', '--algo', type=str, default='gaussian',
                        help='CEM algorithm to use, one of "gaussian" or "nonparametric"')
    parser.add_argument('-r', '--repeat', type=int, default=1,
                        help='Number of times to repeat each action for.')
    parser.add_argument('-e', '--episodes', type=int, default=1,
                        help='Number of episodes to average over.')
    parser.add_argument('-l', '--horizon', type=int, default=30,
                        help='Length of each action sequence to consider.')
    parser.add_argument('-p', '--proposals', type=int, default=1000,
                        help='Number of action sequences to evaluate per iteration.')
    parser.add_argument('-k', '--topk', type=int, default=50,
                        help='Number of best action sequences to refit belief to.')
    parser.add_argument('-i', '--iterations', type=int, default=5,
                        help='Number of optimization iterations for each action sequence.')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Standard deviation of noise for nonparametric version.')
    parser.add_argument('--logdir', type=str, default='runs/cem_gym',
                        help='Tensorboard log directory.')
    parser.add_argument('--save', action='store_true',
                        help='If True, save observations and actions.')
    parser.add_argument('--savedir', type=str, default='save/expert_demonstrations',
                        help='Save directory if save is True.')
    args = parser.parse_args()
    main(args)
