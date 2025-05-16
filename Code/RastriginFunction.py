import math
import numpy as np # Import numpy for array operations
from random import uniform, seed

class RastriginFunction:
    """Class representing the Rastrigin function and its properties"""
    
    def __init__(self):
        self.A = 10
        self.min_bound = -5.12  # Minimum bound of the function
        self.max_bound = 5.12   # Maximum bound of the function
        self.global_minimum = (0, 0)  # 2D global minimum
        self.global_minimum_value = 0
        self.global_minimum_nd = np.zeros(30)  # For n-dimensional cases, the global minimum is at origin (0,0,...,0)
        
    def evaluate_1d(self, x):
        """Evaluate the 1D Rastrigin function at point x"""
        # This version is for scalar x, if x can be an array, use np.cos and np.pi
        return self.A + (x**2 - self.A * math.cos(2 * math.pi * x))
    
    def evaluate_2d(self, x, y):
        """Evaluate the 2D Rastrigin function at point (x, y) or for meshgrid arrays X, Y."""
        # Use np.cos and np.pi for compatibility with numpy arrays from meshgrid
        term_x = x**2 - self.A * np.cos(2 * np.pi * x)
        term_y = y**2 - self.A * np.cos(2 * np.pi * y)
        return self.A * 2 + term_x + term_y
    
    def evaluate(self, x_vector):
        """
        Evaluate the n-dimensional Rastrigin function.
        
        Args:
            x_vector: numpy array of n dimensions
            
        Returns:
            Function value at the given point
        """
        n = len(x_vector)
        sum_term = np.sum(x_vector**2 - self.A * np.cos(2 * np.pi * x_vector))
        return self.A * n + sum_term
    
    def get_random_point(self, random_state=None, dimensions=2):
        """
        Get a random point within the function's bounds.

        Args:
            random_state: Seed for random number generator (optional).
            dimensions: Number of dimensions for the point.

        Returns:
            numpy array of random point within the bounds.
        """
        if random_state is not None:
            np.random.seed(random_state)  # Ensure consistent random points if a seed is provided

        # Generate a random point for the specified number of dimensions
        return np.random.uniform(self.min_bound, self.max_bound, dimensions)
            
    def calculate_local_minima_count(self, dimensions):
        """
        Calculate the theoretical number of local minima for the Rastrigin function.

        Args:
            dimensions: Number of dimensions for which to calculate local minima.

        Returns:
            Approximate number of local minima based on the function's characteristics.
        """
        # Estimate the number of local minima per dimension within the defined bounds
        local_minima_per_dim = math.ceil((self.max_bound - self.min_bound) + 1)
        
        # The total number of local minima increases exponentially with the number of dimensions
        # (local_minima_per_dim)^dimensions
        return local_minima_per_dim ** dimensions