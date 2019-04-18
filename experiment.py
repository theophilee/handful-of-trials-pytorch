import os
import time
import torch
import numpy as np

from utils import Logger


class Experiment:
    def __init__(self, mpc, policy, logdir, savedir, args):
        """Experiment.

        Argument:
            mpc (MPC): Model-predictive controller containing dynamics model
                to be trained.
            policy (Policy): Parameterized reactive policy to be trained by
                imitation learning on model-based controller.
            logdir: Log directory for Tensorboard.
            savedir:
            args (DotMap): A DotMap of experiment parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .task_hor (int): Task horizon.
                .num_rollouts (int): Number of rollouts for which we train.
        """
        self.mpc = mpc
        self.policy = policy
        self.env = args.env
        self.task_hor = args.task_hor
        self.num_rollouts = args.num_rollouts

        self.path_mpc = os.path.join(savedir, 'mpc.pth')
        self.path_policy = os.path.join(savedir, 'policy.pth')

        # Tensorboard summary writer
        self.logger = Logger(logdir)

    def run_experiment(self):
        """Train model and parameterized reactive policy.
        """
        # Initial rollout
        obs, acts, reward_sum = self.sample_rollout(actor='mpc')
        self.mpc.train(obs, acts)

        # Training loop
        for i in range(self.num_rollouts):
            print()
            print("Starting training iteration %d." % (i + 1))

            # Restore model
            self.mpc = torch.load(self.path_mpc)

            # Sample rollout from mpc
            #obs_mpc, acts_mpc, reward_sum_mpc = self.sample_rollout(actor='mpc')

            # Train model
            #metrics_model, weights_model, grads_model = self.mpc.train(obs_mpc, acts_mpc)

            # Save model
            #torch.save(self.mpc, self.path_mpc)

            #print("MPC cumulative reward ", reward_sum_mpc)
            #print("MPC action min {}, max {}, mean {}, std {}".format(
            #    acts_mpc.min(), acts_mpc.max(), acts_mpc.mean(), acts_mpc.std()))

            # Log to Tensorboard
            #step = (i + 1) * self.task_hor
            #self.logger.log_scalar("reward/mpc", reward_sum_mpc, step)

            #for key, metric in metrics_model.items():
            #    self.logger.log_scalar("{}/mean".format(key), metric.mean(), step)

                #for n in range(len(metric)):
                #    self.logger.log_scalar("{}/model{}".format(key, n + 1), metric[n], step)

            #for key, weight in weights_model.items():
            #    self.logger.log_histogram("weight/{}".format(key), weight, step)

            #for key, grad in grads_model.items():
            #    self.logger.log_histogram("grad/{}".format(key), grad, step)

            # Sample rollout from policy and label it with MPC
            obs_policy, acts_policy, reward_sum_policy = self.sample_rollout(actor='policy')
            labels_mpc = self.mpc.label(obs_policy[:-1])

            # Train policy
            #metrics_policy = self.policy.train(obs_mpc[:-1], acts_mpc)
            metrics_policy = self.policy.train(obs_policy[:-1], labels_mpc)

            #self.logger.log_scalar("reward/policy", reward_sum_policy, step)
            #for key, metric in metrics_policy.items():
            #    self.logger.log_scalar(key, metric, step)

            print("Policy cumulative reward ", reward_sum_policy)
            print("Policy action min {}, max {}, mean {}, std {}".format(
                acts_policy.min(), acts_policy.max(), acts_policy.mean(), acts_policy.std()))


    def sample_rollout(self, actor):
        """Sample a rollout.

        Argument:
            actor (str): One of 'mpc', 'policy'.

        Returns:
            obs (numpy.ndarray): Trajectory of observations.
            acts (numpy.ndarray): Trajectory of actions.
            reward_sum (int): Sum of accumulated rewards.
        """
        assert actor in ['mpc', 'policy']
        times, rewards = [], []
        O, A, reward_sum, done = [self.env.reset()], [], 0, False

        if actor == 'mpc':
            self.mpc.reset()

        for t in range(self.task_hor):
            start = time.time()

            if actor == 'mpc':
                A.append(self.mpc.act(O[t]))
            else:
                A.append(self.policy.act(O[t]))

            times.append(time.time() - start)

            obs, reward, done, info = self.env.step(A[t])

            O.append(obs)
            reward_sum += reward

            if done:
                break

        #print("Average action selection time: ", np.mean(times))

        return np.array(O), np.array(A), reward_sum