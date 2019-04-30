import os
import time
import torch
import copy
import numpy as np

import utils
from model_free import TD3


def print_rollout_stats(obs, acts, reward_sum):
    print("Cumulative reward ", reward_sum)
    print("Action min {}, max {}, mean {}, std {}".format(
        acts.min(), acts.max(), acts.mean(), acts.std()))
    print("Obs min {}, max {}, mean {}, std {}".format(
        obs.min(), obs.max(), obs.mean(), obs.std()))


class Experiment:
    def __init__(self, mpc, policy, env_string, logdir, savedir, args):
        """Experiment.

        Arguments:
            mpc (MPC): Model-predictive controller containing dynamics model
                to be trained.
            policy (Policy): Parameterized reactive policy to be trained by
                imitation learning on model-based controller.
            env_string (str): String corresponding to environment.
            logdir: Log directory for Tensorboard.
            savedir: Save directory.
            args (DotMap): A DotMap of experiment parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .task_hor (int): Task horizon.
                .num_rollouts (int): Number of rollouts for which we train.
                .num_imagined_rollouts (int): Number of imagined rollouts per
                    iteration of inner imitation learning loop.
        """
        self.mpc = mpc
        self.policy = policy
        self.env = args.env
        self.task_hor = args.task_hor
        self.num_rollouts = args.num_rollouts
        self.num_imagined_rollouts = args.num_imagined_rollouts

        self.env_string = env_string
        self.savedir = os.path.join(savedir, env_string)
        self.path_mpc = os.path.join(savedir, 'mpc.pth')
        self.path_policy = os.path.join(savedir, 'policy.pth')

        utils.create_directories([self.savedir, logdir])

        # Tensorboard summary writer
        self.logger = utils.Logger(os.path.join(logdir, env_string))

    def run_mpc_baseline(self):
        """Model predictive control baseline (no parameterized policy).
        """
        # Initial random rollout
        obs, acts, reward_sum = self.sample_rollout(actor='mpc')
        self.mpc.train(obs, acts)

        # Training loop
        for i in range(self.num_rollouts):
            print()
            print("Starting training iteration %d." % (i + 1))

            # Sample rollout using mpc
            obs, acts, reward_sum = self.sample_rollout(actor='mpc')
            print_rollout_stats(obs, acts, reward_sum)

            # Train model
            metrics, weights, grads = self.mpc.train(obs, acts)

            # Log to Tensorboard
            step = (i + 1) * self.task_hor
            self.logger.log_scalar("reward/mpc", reward_sum, step)

            for key, metric in metrics.items():
                self.logger.log_scalar("{}/mean".format(key), metric.mean(), step)
            #    for n in range(len(metric)):
            #        self.logger.log_scalar("{}/model{}".format(key, n + 1), metric[n], step)

            #for key, weight in weights.items():
            #    self.logger.log_histogram("weight/{}".format(key), weight, step)
            #for key, grad in grads.items():
            #    self.logger.log_histogram("grad/{}".format(key), grad, step)

            # Save model
            torch.save(self.mpc, self.path_mpc)

    def run_mpc_true_dynamics(self):
        """Model predictive control using true dynamics.
        """
        # TODO super slow -> how to parallellize true dynamics env.step()?
        # Sample rollout using mpc with true dynamics
        obs, acts, reward_sum = self.sample_rollout(actor='mpc_true_dynamics')
        print_rollout_stats(obs, acts, reward_sum)

        # Save expert demonstration
        np.save(os.path.join(self.savedir, 'expert_obs'), obs)
        np.save(os.path.join(self.savedir, 'expert_acts'), acts)

    def run_pretrained_policy(self):
        """Pretrained model-free policy (TD3 algorithm, improvement over DDPG).
        """
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])

        # Restore weights of pretrained policy
        self.TD3 = TD3(state_dim, action_dim, max_action)
        try:
            self.TD3.load(self.env_string, "save/model-free")
        except:
            print("Could not load pretrained policy!")

        # Sample rollouts
        O, A = [], []
        for _ in range(100):
            obs, acts, reward_sum = self.sample_rollout(actor='TD3_pretrained')
            print_rollout_stats(obs, acts, reward_sum)
            O.append(obs)
            A.append(acts)
        O, A = np.array(O), np.array(A)

        # Save rollouts (to use as expert demonstrations)
        np.save(os.path.join(self.savedir, 'expert_obs'), O)
        np.save(os.path.join(self.savedir, 'expert_acts'), A)

    def run_behavior_cloning_basic(self):
        """Train parameterized policy with behaviour cloning on saved demonstrations.
        """
        # Load expert demonstrations
        obs = np.load(os.path.join(self.savedir, 'expert_obs.npy'))
        acts = np.load(os.path.join(self.savedir, 'expert_acts.npy'))
        obs = obs[:, :-1].reshape(-1, self.env.observation_space.shape[0])
        acts = acts.reshape(-1, self.env.action_space.shape[0])

        # Train parameterized policy by behavior cloning
        metrics = self.policy.train(obs, acts, iterative=False)
        print("Imitation learning test error", metrics["policy/mse/test"])
        print("Imitation learning training error", metrics["policy/mse/train"])

        # Sample rollout from parameterized policy for evaluation
        obs_policy, acts_policy, reward_sum_policy = self.sample_rollout(actor='policy')
        print_rollout_stats(obs_policy, acts_policy, reward_sum_policy)

    def run_dagger_basic(self):
        """Train parameterized policy with DAgger using pretrained TD3 expert.
        """
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])

        # Restore weights of pretrained policy
        self.TD3 = TD3(state_dim, action_dim, max_action)
        try:
            self.TD3.load(self.env_string, "save/model-free")
        except:
            print("Could not load pretrained policy!")

        # Sample first rollout from expert and initialize policy
        obs, acts, reward_sum = self.sample_rollout(actor='TD3_pretrained')
        self.policy.train(obs[:-1], acts, iterative=True)

        # Sample subsequent rollouts from policy and label with expert
        for _ in range(50):
            obs, acts, reward_sum = self.sample_rollout(actor='policy')
            print_rollout_stats(obs, acts, reward_sum)

            labels = self.TD3.act_parallel(obs[:-1])
            self.policy.train(obs[:-1], labels, iterative=True)

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

            while True:
                if algo == 'behavior_cloning':
                    # Generate imaginary rollouts with the controller
                    obs_imagined, acts_imagined = self.mpc.sample_imaginary_rollouts(
                        actor='mpc', task_hor=self.task_hor, num_rollouts=self.num_imagined_rollouts)

                if algo == 'dagger':
                    raise NotImplementedError

                    # Generate imaginary rollouts with the policy
                    #obs_imagined, _ = self.mpc.sample_imaginary_rollouts(
                    #    actor='policy', task_hor=self.task_hor, num_rollouts=self.num_imagined_rollouts)

                    # Label rollouts with the controller
                    #acts_imagined = self.mpc.label(obs_imagined[:-1])

                # Train parameterized policy by behavior cloning
                metrics_policy = self.policy.train(obs_imagined.reshape(-1, obs_imagined.shape[-1]),
                                                   acts_imagined.reshape(-1, acts_imagined.shape[-1]))

                # TODO imagined rollouts are completely crazy compared to true rollouts
                # e.g min -7.509242057800293, max 46.9343376159668, mean 5.970691697410477, std 11.511027530038433
                # vs. min - 5.286344230310194, max 10.152343352297112, mean - 0.5799591895719013, std 1.45819241183518

                print("Imitation learning test error", metrics_policy["policy/mse/test"])
                print("Imitation learning training error", metrics_policy["policy/mse/train"])
                print("Imagined rollout obs min {}, max {}, mean {}, std {}".format(
                    obs_imagined.min(), obs_imagined.max(), obs_imagined.mean(), obs_imagined.std()))
                print("Imagined rollout act min {}, max {}, mean {}, std {}".format(
                    acts_imagined.min(), acts_imagined.max(), acts_imagined.mean(), acts_imagined.std()))

                if metrics_policy["policy/mse/test"] < 1.0:
                    break

            # Sample rollout from policy
            obs_policy, acts_policy, reward_sum_policy = self.sample_rollout(actor='policy')

            # Train model
            metrics_model, weights_model, grads_model = self.mpc.train(obs_policy, acts_policy)

            print("Policy cumulative reward ", reward_sum_policy)
            print("True obs min {}, max {}, mean {}, std {}".format(
                obs_policy.min(), obs_policy.max(), obs_policy.mean(), obs_policy.std()))
            print("Policy action min {}, max {}, mean {}, std {}".format(
                acts_policy.min(), acts_policy.max(), acts_policy.mean(), acts_policy.std()))

            # Log to Tensorboard
            step = (i + 1) * self.task_hor
            self.logger.log_scalar("reward/policy", reward_sum_policy, step)

            #for key, metric in metrics_policy.items():
            #    self.logger.log_scalar(key, metric, step)

            for key, metric in metrics_model.items():
                self.logger.log_scalar("{}/mean".format(key), metric.mean(), step)

    def sample_rollout(self, actor):
        """Sample a rollout generated by a given actor in the environment.

        Argument:
            actor (str): One of 'mpc', 'policy', 'mpc_true_dynamics', 'TD3_pretrained'.

        Returns:
            obs (1D numpy.ndarray): Trajectory of observations.
            acts (1D numpy.ndarray): Trajectory of actions.
            reward_sum (int): Sum of accumulated rewards.
        """
        assert actor in ['mpc', 'policy', 'mpc_true_dynamics', 'TD3_pretrained']
        O, A, reward_sum, done, times = [self.env.reset()], [], 0, False, []

        if actor in ['mpc', 'mpc_true_dynamics']:
            self.mpc.reset()

        for t in range(self.task_hor):
            start = time.time()

            if actor == 'mpc':
                A.append(self.mpc.act(O[t]))
            elif actor == 'policy':
                A.append(self.policy.act(O[t]))
            elif actor == 'mpc_true_dynamics':
                A.append(self.mpc.act_true_dynamics(copy.deepcopy(self.env)))
                print(t)
            elif actor == 'TD3_pretrained':
                A.append(self.TD3.act(O[t]))
            else:
                raise NotImplementedError

            times.append(time.time() - start)

            obs, reward, done, info = self.env.step(A[t])

            O.append(obs)
            reward_sum += reward

            if done:
                break

        #print("Average action selection time: ", np.mean(times))

        return np.array(O), np.array(A), reward_sum