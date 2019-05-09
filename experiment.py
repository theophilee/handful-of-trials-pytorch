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
                .env: (OpenAI gym environment) The environment for this agent.
                .num_init_rollouts (int): Number of initial random rollouts.
                .num_rollouts (int): Number of rollouts for which we train.
                .num_imagined_rollouts (int): Number of imagined rollouts per
                    iteration of inner imitation learning loop.
        """
        self.mpc = mpc
        self.policy = policy
        self.env = args.env
        self.task_hor = self.env._max_episode_steps
        self.num_init_rollouts = args.num_init_rollouts
        self.num_rollouts = args.num_rollouts
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
        obs, acts = [], []
        for i in range(self.num_init_rollouts):
            o, a, _ = self.sample_rollout(actor='mpc')
            obs.append(o)
            acts.append(a)
        self.mpc.train(np.array(obs), np.array(acts))

        # Training loop
        for i in range(self.num_rollouts):
            print("\nStarting training iteration %d." % (i + 1))

            # Sample rollout using mpc
            obs, acts, reward_sum = self.sample_rollout(actor='mpc')
            print_rollout_stats(obs, acts, reward_sum)

            # Train model
            metrics = self.mpc.train(obs, acts, iterative=True)

            # Log to Tensorboard
            step = (i + 1) * self.task_hor
            self.logger.log_scalar("reward/mpc", reward_sum, step)
            self.logger.log_scalar("model/mse/test", metrics["model/mse/test"].mean(), step)
            self.logger.log_scalar("model/mse/train", metrics["model/mse/train"].mean(), step)
            self.logger.log_scalar("model/mse/val", metrics["model/mse/val"].mean(), step)

    def run_behavior_cloning_debug(self):
        """Train parameterized policy with behaviour cloning on saved expert demonstrations.
        """
        # Load expert demonstrations
        path = os.path.join('save/mpc_gym_true_dynamics_cmd_line', self.env_str)
        obs = np.load(os.path.join(path, 'obs.npy'))
        acts = np.load(os.path.join(path, 'act.npy'))
        #obs = np.load(f'save/TD3/{self.env_str}_obs.npy')
        #acts = np.load(f'save/TD3/{self.env_str}_act.npy')
        obs = obs[:, :-1].reshape(-1, self.env.observation_space.shape[0])
        acts = acts.reshape(-1, self.env.action_space.shape[0])

        # Train parameterized policy by behavior cloning
        self.policy.train(obs, acts, iterative=False)

        # Sample rollout from parameterized policy for evaluation
        obs_policy, acts_policy, reward_sum_policy = self.sample_rollout(actor='policy')
        print_rollout_stats(obs_policy, acts_policy, reward_sum_policy)

    def run_dagger_debug(self):
        """Train parameterized policy with DAgger.
        """
        # TODO implement action labeling with mpc_true_dynamics?
        pass

    def run_train_model_debug(self):
        """Train dynamics model on saved expert demonstrations.
        """
        # Load expert demonstrations
        path = os.path.join('save/mpc_gym_true_dynamics_cmd_line', self.env_str)
        obs = np.load(os.path.join(path, 'obs.npy'))
        acts = np.load(os.path.join(path, 'act.npy'))

        # Train model iteratively (same setting as real experiment)
        """
        for i, (o, a) in enumerate(zip(obs, acts)):
            print("\nStarting training iteration %d." % (i + 1))

            metrics = self.mpc.train(o, a, iterative=True)
            print('Test', metrics["model/mse/test"].mean().item())
            print('Train', metrics["model/mse/train"].mean().item())
            print('Val', metrics["model/mse/val"].mean().item())
        """
        metrics = self.mpc.train(obs, acts, iterative=False)
        print('Test', metrics["model/mse/test"].mean().item())
        print('Train', metrics["model/mse/train"].mean().item())
        print('Val', metrics["model/mse/val"].mean().item())


    def run_experiment(self, algo):
        """Learn parameterized policy by behavior cloning on trajectories generated by
        model-predictive controller under the approximate model.

        Argument:
            algo (str): one of 'behavior_cloning', 'dagger'
        """
        assert algo in ['behavior_cloning', 'dagger']

        # Initial random rollout
        obs, acts, reward_sum = self.sample_rollout(actor='mpc')
        self.mpc.train(obs, acts)

        # Training loop
        for i in range(self.num_rollouts):
            print("\nStarting training iteration %d." % (i + 1))

            # Reset policy training dataset
            self.policy.reset_training_set()

            # TODO implement this function
            raise NotImplementedError

    def sample_rollout(self, actor):
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

        for t in range(self.task_hor):
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