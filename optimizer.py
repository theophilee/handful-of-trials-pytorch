import torch
from torch.distributions import Normal


class CEMOptimizer:
    def __init__(self, action_space, horizon, popsize, num_elites, iterations):
        """Cross-entropy method optimizer.

        Arguments:
            action_space: OpenAI gym action space.
            horizon (int): Planning horizon.
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the
                distribution at the next iteration.
            iterations (int): The number of iterations to perform during optimization.
        """
        self.act_bound = action_space.high[0]
        self.action_shape = action_space.shape
        self.horizon = horizon
        self.popsize = popsize
        self.num_elites = num_elites
        self.iterations = iterations

        assert num_elites <= popsize, "Number of elites must be at most the population size."

    def obtain_solution(self, start_obs, score_function, particle_info):
        """Optimize multiple CEM planning instances in parallel.

        Arguments:
            start_obs (2D torch.Tensor): Starting observations from which to enroll plans of
                shape (num_obs, obs_features).
            score_function: Function to compute returns of plans.
            particle_info (utils.Metrics): Dictionary to keep track of particle statistics or None.
        """
        num_obs = start_obs.shape[0]
        mean = torch.zeros((num_obs, self.horizon) + self.action_shape)
        std = torch.ones((num_obs, self.horizon) + self.action_shape) * self.act_bound

        for _ in range(self.iterations):
            plans = Normal(mean, std).sample((self.popsize,)).transpose(0, 1)
            scores = score_function(plans.clamp(-self.act_bound, self.act_bound), start_obs, particle_info)
            elites = torch.stack([p[torch.argsort(s)][-self.num_elites:] for p, s in zip(plans, scores)])
            mean, std = torch.mean(elites, dim=1), torch.std(elites, dim=1)

        return mean.clamp(-self.act_bound, self.act_bound)