import os
import time
import numpy as np
import torch

from utils import Logger, Metrics


def print_rollout_stats(obs, acts, length, score):
    print(f"Cumulative reward {score}, episode length {length}")
    print(f"Action min {acts.min()}, max {acts.max()}, mean {acts.mean()}, std {acts.std()}")
    print(f"Obs min {obs.min()}, max {obs.max()}, mean {obs.mean()}, std {obs.std()}")


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
            logdir (str): Log directory for Tensorboard.
            savedir (str): Save directory.
            args (DotMap): A DotMap of experiment parameters.
                .env (OpenAI gym environment): The environment for this agent.
                .expert_demos (bool): If True, add expert demonstrations to
                    initial dynamics model training set.
                .init_steps (int): Number of initial random timesteps.
                .total_steps (int): Number of timesteps for which we train.
                .train_freq (int): Number of timesteps to wait for before
                    retraining model.
                .imaginary_steps (int): Number of imaginary timesteps per
                    inner imitation learning iteration.
        """
        self.mpc = mpc
        self.policy = policy
        self.env = args.env
        self.expert_demos = args.expert_demos
        self.init_steps = args.init_steps
        self.total_steps = args.total_steps
        self.train_freq = args.train_freq
        self.imaginary_steps = args.imaginary_steps

        self.env_str = env_str
        self.savedir = os.path.join(savedir, env_str, param_str)
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        # TensorboardX summary writer
        self.logger = Logger(os.path.join(logdir, f"{env_str}_{param_str}"))

    def run_experiment(self, load_controller, expert_demos):
        """Model predictive control.

        Arguments:
            load_controller (bool): If True, load mpc controller.
            expert_demos (bool): If True, initialize training set with extra expert demonstrations.
        """
        if load_controller:
            self.mpc = torch.load(os.path.join(self.savedir, 'mpc.pth'))

        else:
            # Initial random rollouts
            obs, acts, lengths, _, _ = self._sample_rollouts(self.init_steps, actor=self.mpc)

            if expert_demos:
                obs_expert, acts_expert = self._load_expert_demos()
                obs = obs + tuple(o for o in obs_expert)
                acts = acts + tuple(a for a in acts_expert)

            # Train initial model
            self.mpc.train_initial(obs, acts)

        # Training loop
        step = self.mpc.X.shape[0]
        while step < self.total_steps:
            # Sample rollouts
            start = time.time()
            print(f"Rolling out {self.train_freq} timesteps...")
            obs, acts, lengths, scores, rollouts_metrics = self._sample_rollouts(self.train_freq, actor=self.mpc)
            step += sum(lengths)
            print_rollout_stats(obs[0], acts[0], lengths[0], scores[0])

            act_metrics = Metrics()
            flat_rollouts_metrics = [item for sublist in rollouts_metrics for item in sublist]
            for x in flat_rollouts_metrics:
                act_metrics.store(x)
            for k, v in act_metrics.average().items():
                self.logger.log_scalar(k, v, step)

            self.logger.log_scalar("score/avg_length", np.mean(lengths), step)
            self.logger.log_scalar("score/avg_score", np.mean(scores), step)
            self.logger.log_scalar("time/rollout_time", (time.time() - start), step)

            # Train model
            train_metrics, weights = self.mpc.train_iteration(obs, acts)
            for k, v in train_metrics.items():
                self.logger.log_scalar(k, v, step)
            for k, v in weights.items():
                self.logger.log_histogram(k, v, step)

            # Save model
            torch.save(self.mpc, os.path.join(self.savedir, 'mpc.pth'))

    def _load_expert_demos(self):
        path = os.path.join('save/expert_demonstrations', self.env_str)
        obs = np.load(os.path.join(path, 'obs.npy'))
        acts = np.load(os.path.join(path, 'act.npy'))
        return obs, acts

    def collect_expert_demos(self):
        path = f'/home/theophile/Documents/research/handful-of-trials-pytorch/save/main/{self.env_str}/gaussian_bias'
        self.mpc = torch.load(os.path.join(path, 'mpc.pth'))

        obs, acts, _, _, _ = self._sample_rollouts(self.imaginary_steps, actor=self.mpc)
        np.save(os.path.join(path, 'obs.npy'), obs)
        np.save(os.path.join(path, 'act.npy'), acts)

    def _sample_rollouts(self, max_steps, actor):
        rollouts, steps = [], 0
        while steps < max_steps:
            rollouts.append(self._sample_rollout(actor))
            steps += rollouts[-1][2]
        return zip(*rollouts)

    def _sample_rollout(self, actor, verbose=False):
        """Sample a rollout generated by a given actor in the environment.

        Argument:
            actor: Must provide an act() function operating on 1D np.ndarray.
            verbose: If True, print rollout statistics.

        Returns:
            observations (1D numpy.ndarray): Trajectory of observations (length + 1, obs_features).
            actionss (1D numpy.ndarray): Trajectory of actions (length, act_features).
            length (int): Number of timesteps.
            score (int): Sum of accumulated rewards.
            action_metrics (list[dic]): Side information for each action.
        """
        observations, actions, action_metrics, score = [self.env.reset()], [], [], 0

        for t in range(self.env.max_steps):
            if isinstance(actor, TD3):
                act, act_info = actor.act(observations[t]), {}
            else:
                act, act_info = actor.act(observations[t])
            actions.append(act); action_metrics.append(act_info)
            obs, reward, done, _ = self.env.step(actions[t])
            observations.append(obs)
            score += reward
            if done:
                break

        observations, actions, length = np.array(observations), np.array(actions), t + 1
        if verbose:
            print_rollout_stats(observations, actions, length, score)
        return observations, actions, length, score, action_metrics