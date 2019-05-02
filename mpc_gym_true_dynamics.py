"""
CEM planning on gym environments using the ground truth dynamics.
Example usage:
python mpc_gym_true_dynamics.py halfcheetah run -r 4 -l 12
"""
import argparse
import env # Register environments
import numpy as np
from multiprocessing import Pool
import gym
import copy


# TODO add action repeat
# TODO find optimal CEM hyper-parameters for each environment


def initializer(env):
    global global_env
    global_env = env


def evaluate(actions):
    # Cannot reset state in gym -> copy environment
    env = copy.deepcopy(global_env)
    score = 0
    for action in actions:
        _, reward, _, _ = env.step(action)
        score += reward
    return score


def cem_planner(pool, action_shape, horizon, proposals, topk, iterations):
    mean = np.zeros((horizon,) + action_shape)
    std = np.ones((horizon,) + action_shape)

    for _ in range(iterations):
        plans = [np.random.normal(mean, std) for _ in range(proposals)]
        scores = pool.map(evaluate, plans)
        plans = np.array(plans)[np.argsort(scores)]
        mean, std = plans[-topk:].mean(axis=0), plans[-topk:].std(axis=0)

    return mean[0]


def main(args):
    if args.env == "halfcheetah":
        env = gym.make("MyHalfCheetah-v0")
    elif args.env == "cartpole":
        env = gym.make("MyCartpole-v0")
    else:
        raise NotImplementedError

    scores, durations = [], []
    for _ in range(args.episodes):
        durations.append(0)
        scores.append(0)
        env.reset()

        for _ in range(env._max_episode_steps):
            # Pool of workers, each has its own copy of global environment variable
            pool = Pool(32, initializer, [env])
            action = cem_planner(pool, env.action_space.shape, args.horizon, args.proposals,
                                 args.topk, args.iterations)
            pool.close()
            _, reward, _, _ = env.step(action)
            print(reward)
            durations[-1] += 1
            scores[-1] += reward

    durations = np.array(durations)
    scores = np.array(scores)

    print(durations)
    print(scores)
    print('Mean episode length:', durations.mean())
    print('Mean score:         ', scores.mean())
    print('Standard deviation: ', scores.std())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env',
                        help='Name of the environment to load.')
    parser.add_argument('-r', '--repeat', type=int, default=4,
                        help='Number of times to repeat each action for.')
    parser.add_argument('-e', '--episodes', type=int, default=1,
                        help='Number of episodes to average over.')
    parser.add_argument('-l', '--horizon', type=int, default=25,
                        help='Length of each action sequence to consider.')
    parser.add_argument('-p', '--proposals', type=int, default=500,
                        help='Number of action sequences to evaluate per iteration.')
    parser.add_argument('-k', '--topk', type=int, default=50,
                        help='Number of best action sequences to refit belief to.')
    parser.add_argument('-i', '--iterations', type=int, default=10,
                        help='Number of optimization iterations for each action sequence.')
    args = parser.parse_args()
    main(args)