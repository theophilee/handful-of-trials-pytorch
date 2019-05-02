"""
CEM planning on dm_control tasks using the ground truth dynamics.
Example usage:
python mpc_dm_control_true_dynamics.py cheetah run -r 4 -l 12
"""
import argparse
from dm_control import rl
from dm_control import suite
import numpy as np
from multiprocessing import Pool
from functools import partial


class ActionRepeat(object):
    def __init__(self, env, amount):
        self._env = env
        self._amount = amount
        self._last_action = None
        
    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def step(self, action):
        reward = 0
        discount = 1
        
        for _ in range(self._amount):
            time_step = self._env.step(action)
            reward += time_step.reward
            discount *= time_step.discount
        
            if time_step.last():
                break
        
        time_step = rl.environment.TimeStep(
            step_type=time_step.step_type,
            reward=reward,
            discount=discount,
            observation=time_step.observation)
        
        return time_step

    def reset(self, *args, **kwargs):
        self._last_action = None
        return self._env.reset(*args, **kwargs)


def initializer(env):
    global global_env
    global_env = env


def evaluate(actions, state):
    global_env.reset()
    with global_env.physics.reset_context():
        global_env.physics.data.qpos[:] = state[0]
        global_env.physics.data.qvel[:] = state[1]

    score = 0
    for action in actions:
        time_step = global_env.step(action)
        score += time_step.reward
    return score


def cem_planner(pool, action_spec, state, horizon, proposals, topk, iterations):
    mean = np.zeros((horizon,) + action_spec.shape)
    std = np.ones((horizon,) + action_spec.shape)

    for _ in range(5):
        plans = [np.random.normal(mean, std) for _ in range(proposals)]
        scores = pool.map(partial(evaluate, state=state), plans)
        plans = np.array(plans)[np.argsort(scores)]
        mean, std = plans[-topk:].mean(axis=0), plans[-topk:].std(axis=0)
        
    return mean[0]


def main(args):
    env = suite.load(args.domain, args.task)
    env = ActionRepeat(env, args.repeat)

    # Pool of workers, each has its own copy of global environment variable
    pool = Pool(32, initializer, [env])

    scores, durations = [], []
    for _ in range(args.episodes):
        durations.append(0)
        scores.append(0)
        time_step = env.reset()

        while not time_step.last():
            state = (env.physics.data.qpos, env.physics.data.qvel)
            action = cem_planner(pool, env.action_spec(), state, args.horizon,
                                 args.proposals, args.topk, args.iterations)
            print(action)
            time_step = env.step(action)
            durations[-1] += 1
            scores[-1] += time_step.reward

    durations = np.array(durations)
    scores = np.array(scores)
    
    print(durations)
    print(scores)
    print('Mean episode length:', durations.mean())
    print('Mean score:         ', scores.mean())
    print('Standard deviation: ', scores.std())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('domain',
                        help='Name of the environment to load.')
    parser.add_argument('task',
                        help='Name of the task to load.')
    parser.add_argument('-r', '--repeat', type=int, default=4, 
                        help='Number of times to repeat each action for.')
    parser.add_argument('-e', '--episodes', type=int, default=1, 
                        help='Number of episodes to average over.')
    parser.add_argument('-l', '--horizon', type=int, default=12, 
                        help='Length of each action sequence to consider.')
    parser.add_argument('-p', '--proposals', type=int, default=1000, 
                        help='Number of action sequences to evaluate per iteration.')
    parser.add_argument('-k', '--topk', type=int, default=100,
                        help='Number of best action sequences to refit belief to.')
    parser.add_argument('-i', '--iterations', type=int, default=10,
                        help='Number of optimization iterations for each action sequence.')
    args = parser.parse_args()
    main(args)