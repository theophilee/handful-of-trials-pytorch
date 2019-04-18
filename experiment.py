import time
import numpy as np

from utils import Logger


class Experiment:
    def __init__(self, policy, logdir, args):
        """Experiment.

        Argument:
            policy (MPC): Policy to be trained.
            logdir: Log directory for Tensorboard.
            args (DotMap): A DotMap of experiment parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .task_hor (int): Task horizon.
                .num_rollouts (int): Number of rollouts for which we train.
        """
        self.policy = policy
        self.env = args.env
        self.task_hor = args.task_hor
        self.num_rollouts = args.num_rollouts

        # Tensorboard summary writer
        self.logger = Logger(logdir)

    def run_experiment(self):
        """Train policy.
        """
        # Initial rollout
        obs, acts, reward_sum = self.sample_rollout()
        self.policy.train(obs, acts)

        # Training loop
        for i in range(self.num_rollouts):
            print("Starting training iteration %d." % (i + 1))

            # Sample rollout
            obs, acts, reward_sum = self.sample_rollout()

            # Train model
            metrics, weights, grads = self.policy.train(obs, acts)

            # Log to Tensorboard
            step = (i + 1) * self.task_hor
            self.logger.log_scalar("reward", reward_sum, step)

            for key, metric in metrics.items():
                self.logger.log_scalar("{}/mean".format(key), metric.mean(), step)

                for n in range(len(metric)):
                    self.logger.log_scalar("{}/model{}".format(key, n + 1), metric[n], step)

            for key, weight in weights.items():
                self.logger.log_histogram("weight/{}".format(key), weight, step)

            for key, grad in grads.items():
                self.logger.log_histogram("grad/{}".format(key), grad, step)


    def sample_rollout(self):
        """Sample a rollout.

        Returns:
            obs (numpy.ndarray): Trajectory of observations.
            acts (numpy.ndarray): Trajectory of actions.
            reward_sum (int): Sum of accumulated rewards.
        """
        times, rewards = [], []
        O, A, reward_sum, done = [self.env.reset()], [], 0, False

        self.policy.reset()
        for t in range(self.task_hor):
            start = time.time()
            A.append(self.policy.act(O[t]))
            times.append(time.time() - start)

            obs, reward, done, info = self.env.step(A[t])

            O.append(obs)
            reward_sum += reward

            if done:
                break

        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))
        print("Cumulative reward: ", reward_sum)

        return np.array(O), np.array(A), reward_sum