import env # Register environments
import numpy as np
from multiprocessing import Pool
from functools import partial
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle
import gym


ENV = 'MySwimmer-v2'


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
            obs, reward, _, _ = self._env.step(action)
            total_reward += reward

        return obs, total_reward, False, {}

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
        _, reward, _, _ = global_env.step(action)
        score += reward
    return score


def cem_planner(pool, action_space, state, horizon, proposals, topk, iterations):
    action_bound = action_space.high[0]
    mean = np.zeros((horizon,) + action_space.shape)
    std = np.ones((horizon,) + action_space.shape) * action_bound

    for _ in range(iterations):
        plans = np.random.normal(mean, std, size=(proposals,) + mean.shape)
        scores = pool.map(partial(evaluate, state=state), plans.clip(-action_bound, action_bound))
        plans = plans[np.argsort(scores)]
        mean, std = plans[-topk:].mean(axis=0), plans[-topk:].std(axis=0)

    return mean[0].clip(-action_bound, action_bound)


def objective(space):
    env = gym.make(ENV)
    env = ActionRepeat(env, int(space['repeat']))

    proposals = 1000
    iterations = 10

    # Pool of workers, each has its own copy of global environment variable
    pool = Pool(32, initializer, [env])

    cost = 0
    env.reset()
    for _ in range(env.num_steps):
        state = env.sim.get_state()
        action = cem_planner(pool, env.action_space, state, int(space['horizon']),
                             proposals, int(space['topk']), iterations)
        _, reward, _, _ = env.step(action)
        cost -= reward
    return {'loss': cost, 'status': STATUS_OK}


def run_trials(space, objective, init, step, filename):
    try:
        # Try to load an already saved trials object, and increase the max
        trials = pickle.load(open("%s.hyperopt" % filename, "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, step))
    except:
        # Create new trials object and start searching
        trials = Trials()
        max_trials = init

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_trials,
                trials=trials)

    print("Best: ", best)

    # Save trials object
    with open("%s.hyperopt" % filename, "wb") as f:
        pickle.dump(trials, f)


if __name__ == '__main__':
    """
    Objective:
        'MyCartpole-v0': 178
        'MySwimmer-v2': 360
        'MyHalfCheetah-v2': 15000
        
    Current best:
        'MyCartpole-v0' {'horizon': 12, 'repeat': 4, 'topk': 50} -> 178 (std = 3 for 10 runs)
        'MySwimmer-v2' {'horizon': 17, 'repeat': 4, 'topk': 30} -> 312
        'MyHalfCheetah-v2' {'horizon': 11, 'repeat': 4, 'topk': 40} -> 13907 (std 1541 for 20 runs)
    """
    #space = {'repeat': hp.quniform('repeat', 2, 8, 1),
    #         'horizon': hp.quniform('horizon', 10, 25, 1),
    #         'topk': hp.quniform('topk', 10, 90, 10)}
    space = {'repeat': hp.quniform('repeat', 2, 5, 1),
             'horizon': hp.quniform('horizon', 10, 20, 1),
             'topk': hp.quniform('topk', 20, 80, 10)}

    while True:
        run_trials(space, objective, 2, 2, ENV)