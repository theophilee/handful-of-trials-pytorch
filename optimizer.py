import numpy as np
import scipy.stats as stats


class CEMOptimizer:
    def __init__(self, sol_dim, max_iters, popsize, num_elites, cost_function,
                 upper_bound, lower_bound, epsilon=0.001, alpha=0.25):
        """Cross-entropy method optimizer.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
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
        self.sol_dim = sol_dim
        self.max_iters = max_iters
        self.popsize = popsize
        self.num_elites = num_elites
        self.cost_function = cost_function
        self.upper_bound, self.lower_bound, = upper_bound, lower_bound
        self.epsilon = epsilon
        self.alpha = alpha

        assert num_elites <= popsize, "Number of elites must be at most the population size."

    def obtain_solution(self, init_mean, init_var):
        """Optimizes the cost function using the provided initial candidate distribution.

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        mean, var, t = init_mean, init_var, 0
        distrib = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            # Constrain variance
            var_upper_bound = np.minimum(np.square((mean - self.lower_bound) / 2),
                                         np.square((self.upper_bound - mean) / 2))
            var = np.minimum(var, var_upper_bound)

            # Sample population
            samples = distrib.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(var) + mean
            samples = samples.astype(np.float32)

            # Select elites
            costs = self.cost_function(samples)
            elites = samples[np.argsort(costs)][:self.num_elites]

            # Update distribution
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1

        return mean
