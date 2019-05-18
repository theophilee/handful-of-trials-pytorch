import os
import time
import numpy as np
import torch

from utils import Logger


def print_rollout_stats(obs, acts, score):
    print("Cumulative reward ", score)
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
                .init_rollouts (int): Number of initial random rollouts.
                .total_rollouts (int): Number of rollouts for which we train.
                .train_freq (int): Number of episodes to wait for before
                    retraining model.
                .imaginary_rollouts (int): Number of imaginary rollouts per
                    inner imitation learning iteration.
        """
        self.mpc = mpc
        self.policy = policy
        self.env = args.env
        self.expert_demos = args.expert_demos
        self.init_rollouts = args.init_rollouts
        self.total_rollouts = args.total_rollouts
        self.train_freq = args.train_freq
        self.imaginary_rollouts = args.imaginary_rollouts

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
        obs, acts, _ = self._sample_rollouts(self.init_rollouts, actor=self.mpc)

        # Optionally add expert demonstrations
        if self.expert_demos:
            obs_expert, acts_expert = self._load_expert_demos()
            obs = np.concatenate((obs, obs_expert), axis=0)
            acts = np.concatenate((acts, acts_expert), axis=0)

        # Train initial model
        self.mpc.train(obs, acts, iterative=True)

        # Training loop
        step = self.init_rollouts
        while step < self.total_rollouts:
            step += self.train_freq

            # Sample rollouts
            start = time.time()
            print(f"Rolling out {self.train_freq} trajectories...")
            obs, acts, avg_score = self._sample_rollouts(self.train_freq, actor=self.mpc)
            print_rollout_stats(obs, acts, avg_score)
            self.logger.log_scalar("rollout/avg_score", avg_score, step)
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

        # Train parameterized policy by behavior cloning
        self.policy.train(obs[:, :-1], acts, iterative=False)

        # Sample rollout from parameterized policy for evaluation
        obs_policy, acts_policy, score_policy = self._sample_rollout(actor=self.policy)
        print_rollout_stats(obs_policy, acts_policy, score_policy)

        torch.save(self.policy, os.path.join(self.savedir, 'policy.pth'))

    def run_train_model_debug(self):
        """Train dynamics model on saved expert demonstrations.
        """
        #obs, acts, _ = self._sample_rollouts(self.init_rollouts, actor=self.mpc)
        obs, acts = self._load_expert_demos()
        #obs = np.concatenate((obs, obs_expert), axis=0)
        #acts = np.concatenate((acts, acts_expert), axis=0)

        metrics, _ = self.mpc.train(obs, acts, iterative=False, debug_logger=self.logger)
        for k, v in metrics.items():
            print(f'{k}: {v}')

        torch.save(self.mpc, os.path.join(self.savedir, 'model_expert.pth'))
        #torch.save(self.mpc, os.path.join(self.savedir, 'model_random.pth'))

    def run_experiment_debug(self):
        """Train parameterized policy by imitation learning (DAgger) on trajectories
        generated under the learned dynamics model, using pretrained dynamics model.
        """
        self.run_behavior_cloning_debug()
        self.run_train_model_debug()

        # Load pretrained model
        self.mpc = torch.load(os.path.join(self.savedir, 'model_expert.pth'))
        #self.mpc = torch.load(os.path.join(self.savedir, 'model_random.pth'))

        # Load pretrained expert
        expert = torch.load(os.path.join(self.savedir, 'policy.pth'))
        obs_expert, acts_expert, score_expert = self._sample_rollout(actor=expert)
        print("Expert performance")
        print_rollout_stats(obs_expert, acts_expert, score_expert)

        # Train parameterized policy by imitating expert under learned model
        for i in range(self.imaginary_rollouts):
            # Train
            obs, acts, score = self.mpc.sample_imaginary_rollouts(1, actor=self.policy)
            obs = obs.view([-1, obs.shape[-1]])
            labels = expert.act_parallel(obs)
            metrics = self.policy.train(obs, labels, iterative=True)
            print(f"Validation loss after training policy {metrics['policy/mse/val']}")

            # Evaluate
            obs_eval, acts_eval, score_eval = self._sample_rollout(actor=self.policy)

            print(f"\nTraining iteration {i}")
            print("Imaginary rollout statistics")
            print_rollout_stats(obs, acts, score)
            print("Evaluation statistics")
            print_rollout_stats(obs_eval, acts_eval, score_eval)

        torch.save(self.policy, os.path.join(self.savedir, 'policy_imitation.pth'))
        """
        for _ in range(20):
            expert = torch.load(os.path.join(self.savedir, 'policy.pth'))
            imitator = torch.load(os.path.join(self.savedir, 'policy_imitation.pth'))
            obs_expert, acts_expert, score_expert = self._sample_rollout(actor=expert)
            obs_imitation, acts_imitation, score_imitation = self._sample_rollout(actor=imitator)
            print_rollout_stats(obs_expert, acts_expert, score_expert)
            print_rollout_stats(obs_imitation, acts_imitation, score_imitation)
            print()
        """

    def run_experiment(self):
        """Train parameterized policy by imitation learning (DAgger) on trajectories
        generated under the learned dynamics model in the inner loop of model optimization.
        """
        # Initial random rollouts
        obs, acts, _ = self._sample_rollouts(self.init_rollouts, actor=self.mpc)

        # Optionally add expert demonstrations
        if self.expert_demos:
            obs_expert, acts_expert = self._load_expert_demos()
            obs = np.concatenate((obs, obs_expert), axis=0)
            acts = np.concatenate((acts, acts_expert), axis=0)

        # Train initial model
        self.mpc.train(obs, acts, iterative=True)

        # Training loop
        step = self.init_rollouts * 10
        while step < self.total_rollouts:
            step += self.train_freq * 10

            # Sample rollouts from policy
            start = time.time()
            obs, acts, score = self._sample_rollouts(self.train_freq, actor=self.policy)
            self.logger.log_scalar("rollout/score", score, step)
            self.logger.log_scalar("rollout/avg_time", (time.time() - start) / self.train_freq, step)

            # Train model
            metrics, tensors = self.mpc.train(obs, acts, iterative=True)
            for k, v in metrics.items():
                self.logger.log_scalar(k, v, step)
            for k, v in tensors.items():
                self.logger.log_histogram(k, v, step)

            # Train policy with DAgger in inner loop
            self.policy.reset_training_set()
            for i in range(self.imaginary_rollouts):
                start = time.time()
                obs, acts, score = self.mpc.sample_imaginary_rollouts(1, actor=self.policy)
                obs = obs.view([-1, obs.shape[-1]])
                labels = self.mpc.act_parallel(obs)
                self.policy.train(obs, labels, iterative=True)
                _, _, score_eval = self._sample_rollout(actor=self.policy)
                self.logger.log_scalar("imaginary_rollout/score_imaginary", score, step + i)
                self.logger.log_scalar("imaginary_rollout/score_eval", score_eval, step + i)
                self.logger.log_scalar("imaginary_rollout/time", time.time() - start, step + i)

    def _load_expert_demos(self):
        path = os.path.join('save/expert_demonstrations', self.env_str)
        obs = np.load(os.path.join(path, 'obs.npy'))
        acts = np.load(os.path.join(path, 'act.npy'))
        return obs, acts

    def _sample_rollouts(self, num, actor):
        observations, actions, scores = [], [], []
        for _ in range(num):
           obs, acts, score = self._sample_rollout(actor)
           observations.append(obs); actions.append(acts), scores.append(score)
        return np.array(observations), np.array(actions), np.array(scores).mean()

    def _sample_rollout(self, actor):
        """Sample a rollout generated by a given actor in the environment.

        Argument:
            actor: Must provide an act() function operating on 1D np.ndarray.

        Returns:
            obs (1D numpy.ndarray): Trajectory of observations.
            acts (1D numpy.ndarray): Trajectory of actions.
            score (int): Sum of accumulated rewards.
        """
        observations, actions, score, times = [self.env.reset()], [], 0, []
        for t in range(self.env.num_steps):
            actions.append(actor.act(observations[t]))
            obs, reward, _, _ = self.env.step(actions[t])
            observations.append(obs)
            score += reward
        return np.array(observations), np.array(actions), score