import os
import time
import numpy as np

from utils import Logger


def print_rollout_stats(obs, acts, reward_sum):
    print("Cumulative reward ", reward_sum)
    print("Action min {}, max {}, mean {}, std {}".format(
        acts.min(), acts.max(), acts.mean(), acts.std()))
    print("Obs min {}, max {}, mean {}, std {}".format(
        obs.min(), obs.max(), obs.mean(), obs.std()))


class Experiment:
    def __init__(self, mpc, policy, env_str, param_str, logdir, savedir, args):
        """Experiment.

        Arguments:
            mpc (MPC): Model-predictive controller containing dynamics model
                to be trained.
            policy (Policy): Parameterized reactive policy to be trained by
                imitation learning on model-based controller.
            env_str (str): String descriptor of environment.
            param_str (str): String descriptor of experiment hyper-parameters.
            logdir: Log directory for Tensorboard.
            savedir: Save directory.
            args (DotMap): A DotMap of experiment parameters.
                .env (OpenAI gym environment): The environment for this agent.
                .expert_demos (bool): If True, add expert demonstrations to
                    initial dynamics model training set.
                .num_init_rollouts (int): Number of initial random rollouts.
                .num_rollouts (int): Number of rollouts for which we train.
                .train_freq (int): Number of episodes to wait for before
                    retraining model.
                .num_imagined_rollouts (int): Number of imagined rollouts per
                    iteration of inner imitation learning loop.
        """
        self.mpc = mpc
        self.policy = policy
        self.env = args.env
        self.expert_demos = args.expert_demos
        self.num_init_rollouts = args.num_init_rollouts
        self.num_rollouts = args.num_rollouts
        self.train_freq = args.train_freq
        self.num_imagined_rollouts = args.num_imagined_rollouts

        self.env_str = env_str
        self.savedir = os.path.join(savedir, env_str)
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        # TensorboardX summary writer
        self.logger = Logger(os.path.join(logdir, "{}_{}".format(env_str, param_str)))

    def run_mpc_baseline(self):
        """Model predictive control baseline, no parameterized policy.
        """
        # Initial random rollouts
        obs, acts, _ = self._sample_rollouts(self.num_init_rollouts, actor='mpc')

        # Optionally add expert demonstrations
        if self.expert_demos:
            obs_expert, acts_expert = self._load_expert_demos()
            obs = np.concatenate((obs, obs_expert), axis=0)
            acts = np.concatenate((acts, acts_expert), axis=0)

        # Train initial model
        self.mpc.train(obs, acts, iterative=True)

        # Training loop
        step = self.num_init_rollouts
        while step < self.num_rollouts:
            step += self.train_freq

            # Sample rollouts
            start = time.time()
            print(f"Rolling out {self.train_freq} trajectories...")
            obs, acts, avg_return = self._sample_rollouts(self.train_freq, actor='mpc')
            self.logger.log_scalar("rollout/avg_reward", avg_return, step)
            self.logger.log_scalar("rollout/avg_time", (time.time() - start) / self.train_freq, step)

            # Train model
            metrics, tensors = self.mpc.train(obs, acts, iterative=True)
            for k, v in metrics.items():
                self.logger.log_scalar(k, v, step)
            for k, v in tensors.items():
                self.logger.log_histogram(k, v, step)

    def run_behavior_cloning_debug(self):
        """Train parameterized policy with behaviour cloning on saved expert demonstrations.
        """
        # Load expert demonstrations
        obs, acts = self._load_expert_demos()
        #obs = np.load(f'save/TD3/{self.env_str}_obs.npy')
        #acts = np.load(f'save/TD3/{self.env_str}_act.npy')
        obs = obs[:, :-1].reshape(-1, self.env.observation_space.shape[0])
        acts = acts.reshape(-1, self.env.action_space.shape[0])

        # Train parameterized policy by behavior cloning
        self.policy.train(obs, acts, iterative=False)

        # Sample rollout from parameterized policy for evaluation
        obs_policy, acts_policy, reward_sum_policy = self._sample_rollout(actor='policy')
        print_rollout_stats(obs_policy, acts_policy, reward_sum_policy)

    def run_train_model_debug(self):
        """Train dynamics model on saved expert demonstrations.
        """
        #obs, acts = self._load_expert_demos()
        obs, acts, _ = self._sample_rollouts(self.num_init_rollouts, actor='mpc')

        metrics, _ = self.mpc.train(obs, acts, iterative=False, debug_logger=self.logger)
        for k, v in metrics.items():
            print(f'{k}: {v}')

    def run_experiment(self, algo):
        """Learn parameterized policy by behavior cloning on trajectories generated by
        model-predictive controller under the approximate model.

        Argument:
            algo (str): one of 'behavior_cloning', 'dagger'
        """
        assert algo in ['behavior_cloning', 'dagger']
        raise NotImplementedError
        # TODO implement this function

    def _load_expert_demos(self):
        path = os.path.join('save/mpc_gym_true_dynamics_cmd_line', self.env_str)
        obs = np.load(os.path.join(path, 'obs.npy'))
        acts = np.load(os.path.join(path, 'act.npy'))
        return obs, acts

    def _sample_rollouts(self, num, actor):
       observations, actions, returns = [], [], []
       for _ in range(num):
           obs, acts, ret = self._sample_rollout(actor)
           observations.append(obs); actions.append(acts), returns.append(ret)
       return np.array(observations), np.array(actions), np.array(returns).mean()

    def _sample_rollout(self, actor):
        """Sample a rollout generated by a given actor in the environment.

        Argument:
            actor (str): One of 'mpc', 'policy'.

        Returns:
            obs (1D numpy.ndarray): Trajectory of observations.
            acts (1D numpy.ndarray): Trajectory of actions.
            reward_sum (int): Sum of accumulated rewards.
        """
        assert actor in ['mpc', 'policy']
        observations, actions, reward_sum, times = [self.env.reset()], [], 0, []

        for t in range(self.env.num_steps):
            start = time.time()
            if actor == 'mpc':
                actions.append(self.mpc.act(observations[t]))
            elif actor == 'policy':
                actions.append(self.policy.act(observations[t]))
            times.append(time.time() - start)
            obs, reward, _, _ = self.env.step(actions[t])
            observations.append(obs)
            reward_sum += reward

        #print("Average action selection time: ", np.mean(times))
        return np.array(observations), np.array(actions), reward_sum