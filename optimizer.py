import numpy as np
import scipy.stats as stats


class CEMOptimizer:
    def __init__(self, max_iters, popsize, num_elites, cost_function, upper_bound,
                 lower_bound, epsilon=0.001, alpha=0.25):
        """Cross-entropy method optimizer.

        Arguments:
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the
                distribution at the next iteration.
            upper_bound (np.ndarray): An array of upper bounds
            lower_bound (np.ndarray): An array of lower bounds
            epsilon (float): If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for
                the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean
        """
        self.max_iters = max_iters
        self.popsize = popsize
        self.num_elites = num_elites
        self.cost_function = cost_function
        self.upper_bound, self.lower_bound, = upper_bound, lower_bound
        self.init_var = np.square(self.upper_bound - self.lower_bound) / 16
        self.sol_dim = self.init_var.shape[0]
        self.epsilon = epsilon
        self.alpha = alpha

        assert num_elites <= popsize, "Number of elites must be at most the population size."

    def obtain_solution(self, init_mean):
        """ Optimize multiple CEM problems in parallel (parallel cost function computation)
        using the provided initial candidate distributions.

        Arguments:
            init_mean (2D np.ndarray): The means of the initial candidate distributions of
                shape (num_problems, sol_dim).
        """
        mean, var, t = init_mean, self.init_var, 0
        num_problems = init_mean.shape[0]
        distrib = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            # Constrain variance
            var_upper_bound = np.minimum(np.square((mean - self.lower_bound) / 2),
                                         np.square((self.upper_bound - mean) / 2))
            var = np.minimum(var, var_upper_bound)

            # Sample population and reshape it to (num_problems, popsize, sol_dim)
            samples = distrib.rvs(size=[self.popsize, *init_mean.shape]) * np.sqrt(var) + mean
            samples = samples.astype(np.float32).transpose(1, 0, 2)

            # Compute costs for all problems in parallel of shape (num_problems, popsize)
            costs = self.cost_function(samples)

            # Select elites for each problem independently
            elites = []
            for i in range(num_problems):
                elites.append(samples[i, np.argsort(costs[i])[:self.num_elites]])
            elites = np.array(elites)

            # Update distributions
            new_mean = np.mean(elites, axis=1)
            new_var = np.var(elites, axis=1)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1

        return mean