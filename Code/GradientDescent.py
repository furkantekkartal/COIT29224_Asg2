"""
@author: 12223508

Implementation of the Gradient Descent algorithm for optimizing the Rastrigin function.
This serves as a baseline algorithm to compare with evolutionary strategies.
Gradient Descent works by iteratively moving in the direction of steepest descent (negative gradient).

Key features:
- Uses the gradient of the function to determine the update direction
- Implements a simple learning rate decay schedule
- Constrains solutions to stay within function boundaries
- Supports optimization in different dimensions (1D, 2D, 30D, etc.)
"""

import math
import numpy as np
from RastriginFunction import RastriginFunction

class GradientDescent:
    """
    Implementation of the Gradient Descent optimization algorithm.
    
    Gradient Descent is a first-order iterative optimization algorithm for finding a local minimum of a 
    differentiable function. The algorithm takes steps proportional to the negative of the gradient of 
    the function at the current point.
    
    For multimodal functions like Rastrigin, gradient descent is likely to get stuck in local minima,
    making it a good baseline to compare with more advanced methods like Evolution Strategies.
    """
    
    def __init__(self, function, start_point=None, random_seed=None, learning_rate=0.01, objective=0, max_generations=10000, dimensions=2):
        # Set random seed if provided, to ensure consistent start if start_point is None
        if random_seed is not None:
            np.random.seed(random_seed)     # Set NumPy's random seed

        self.dimensions = dimensions        # Store the number of dimensions
        
        # Initialize starting point
        if start_point is None:
            if hasattr(function, 'get_random_point'):
                self.position = np.array(function.get_random_point(dimensions=dimensions))  # Pass dimensions directly
            else:
                # Generate random points within the function's bounds
                min_bound = function.min_bound if hasattr(function, 'min_bound') else -5.12
                max_bound = function.max_bound if hasattr(function, 'max_bound') else 5.12
                self.position = np.random.uniform(min_bound, max_bound, dimensions)
        else:
            # Use provided start point, ensuring it's a numpy array
            if len(start_point) != dimensions:
                raise ValueError(f"start_point must have {dimensions} dimensions")
            self.position = np.array(start_point)

        self.start_point = self.position.copy()                    # Store initial point for reporting
        self.function = function                                   # The function to be optimized
        self.num_parameters = dimensions                           # Number of dimensions for optimization
        self.learning_rate = learning_rate                         # Initial learning rate
        self.objective = objective                                 # Target objective value
        self.max_error = 0.001                                     # Tolerance for stopping criterion
        self.max_generations = max_generations                     # Maximum number of generations to run
        self.end_point = None                                      # Will store final solution
        self.initial_score = self.evaluate_position(self.position) # Calculate score at starting point
        self.result = self.initial_score                           # Current best score
        self.error = abs(self.objective - self.result)             # Distance from target objective
        self.generations_run = 0                                   # Counter for generations actually run
        self.best_position = self.position.copy()                  # Track best solution found
        self.best_score = self.initial_score                       # Track best score
        self.stagnation_counter = 0                                # Counter for stagnation detection

    def evaluate_position(self, position):
        """Evaluate the fitness of a position based on the number of dimensions"""
        if self.dimensions == 1:
            return self.function.evaluate_1d(position[0]) if hasattr(self.function, 'evaluate_1d') else self.function.evaluate(position)
        elif self.dimensions == 2:
            return self.function.evaluate_2d(position[0], position[1]) if hasattr(self.function, 'evaluate_2d') else self.function.evaluate(position)
        else:
            return self.function.evaluate(position) if hasattr(self.function, 'evaluate') else sum([self.function.evaluate_1d(x) for x in position])

    def calculate_error(self):
        """Calculate the error as the absolute difference from the objective"""
        self.error = abs(self.objective - self.result)
        return self.error

    def calculate_gradient(self, position, epsilon=1e-6):
        """
        Calculate the gradient of the function at a given position using finite differences.
        
        Args:
            position: Point at which to calculate the gradient
            epsilon: Small step size for finite difference approximation
            
        Returns:
            numpy array representing the gradient vector
        """
        gradient = np.zeros(self.dimensions)
        
        for i in range(self.dimensions):
            # Create two nearby points differing only in dimension i
            pos_plus = position.copy()
            pos_minus = position.copy()
            
            pos_plus[i] += epsilon
            pos_minus[i] -= epsilon
            
            # Evaluate function at both points
            f_plus = self.evaluate_position(pos_plus)
            f_minus = self.evaluate_position(pos_minus)
            
            # Calculate partial derivative using central difference
            gradient[i] = (f_plus - f_minus) / (2 * epsilon)
            
        return gradient

    def evolve(self):
        """
        Run the gradient descent optimization algorithm.
        
        The algorithm iteratively follows the negative gradient of the function,
        taking steps proportional to the learning rate. For multimodal functions,
        it includes stagnation detection and learning rate decay to attempt to
        find better solutions.
        """
        count = 0                    # Generation counter
        min_learning_rate = 1e-6     # Minimum learning rate to prevent too small steps
        decay_factor = 0.95          # Learning rate decay factor
        decay_interval = 100         # Interval for learning rate decay
        improvement_threshold = 1e-6 # Threshold for detecting stagnation
        max_stagnation = 50          # Maximum stagnation count before reducing learning rate
        
        # Define bounds
        min_bound = self.function.min_bound if hasattr(self.function, 'min_bound') else -5.12
        max_bound = self.function.max_bound if hasattr(self.function, 'max_bound') else 5.12
        
        last_score = self.result
        
        # Print initial status
        print(f"  Starting optimization with learning rate: {self.learning_rate:.6f}, initial score: {self.result:.6f}")
        
        while count < self.max_generations and self.error > self.max_error:
            # Calculate gradient at current position
            gradient = self.calculate_gradient(self.position)
            
            # Update position using gradient descent
            new_position = self.position - self.learning_rate * gradient
            
            # Ensure we stay within bounds
            new_position = np.clip(new_position, min_bound, max_bound)
            
            # Evaluate new position
            new_score = self.evaluate_position(new_position)
            
            # Update position and score if improved
            if new_score < self.result:
                self.position = new_position
                self.result = new_score
                
                # Update best position if this is the best score so far
                if new_score < self.best_score:
                    self.best_position = new_position.copy()
                    self.best_score = new_score
                
                # Reset stagnation counter if significant improvement
                if last_score - new_score > improvement_threshold:
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1
            else:
                # No improvement, increase stagnation counter
                self.stagnation_counter += 1
            
            # Log progress only occasionally
            if count % 1000 == 0:
                print(f"  Generation {count}: Learning rate: {self.learning_rate:.6f}, best score: {self.best_score:.6f}")
            
            # Adjust learning rate periodically or when stagnation is detected
            if count % decay_interval == 0 or self.stagnation_counter >= max_stagnation:
                self.learning_rate = max(min_learning_rate, self.learning_rate * decay_factor)
                self.stagnation_counter = 0
                
                # If learning rate is at minimum and we're still not improving,
                # try a random restart from the best position with a perturbed location
                if self.learning_rate == min_learning_rate and count % (decay_interval * 5) == 0:
                    print(f"    -Attempting random restart at generation {count}")
                    # Perturb the best position found so far
                    perturbation = np.random.uniform(-0.5, 0.5, self.dimensions)
                    self.position = np.clip(self.best_position + perturbation, min_bound, max_bound)
                    self.result = self.evaluate_position(self.position)
                    self.learning_rate = self.learning_rate * 0.1       # Reduced learning rate for fine-tuning
            
            last_score = self.result
            self.calculate_error()
            count += 1
            
        # After search is complete, use the best solution found
        self.position = self.best_position.copy()
        self.result = self.best_score
        self.generations_run = count
        self.end_point = self.best_position.copy()
        self.calculate_error()  # Final error calculation
        
        # Print final status
        print(f"  Optimization completed after {count} generations with final score: {self.result:.6f}")