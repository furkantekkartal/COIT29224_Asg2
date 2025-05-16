"""
@author: 12223508

Implementation of the (1+1)-Evolution Strategy with 1/5 Success Rule for optimizing the Rastrigin function.
This algorithm works by maintaining a single solution and generating a single offspring in each generation.
The offspring replaces the parent if it has a better fitness. The step size (sigma) is adapted according
to the 1/5 success rule to balance exploration and exploitation during the search.

Key features:
- Uses a self-adaptive mutation step size based on the 1/5 success rule
- Implements restart mechanism to escape from local optima
- Applies independent Gaussian mutations to each coordinate
- Constrains solutions to stay within the function's boundaries
- Supports optimization in different dimensions (1D, 2D, 30D, etc.)
"""

import math
from random import uniform, seed, gauss
import numpy as np
from RastriginFunction import RastriginFunction
# Constants that can be shared across optimization algorithms
ADAPT_CONST = 0.5  # Adaptation constant for 1/5 success rule
                     ## Higher values cause gentler adaptation of sigma
                     ## Lower values cause more aggressive adaptation of sigma

class OnePlusOneESFifthRule:
    """
    Implementation of (1+1)-Evolution Strategy optimization algorithm with 1/5 success rule.
    
    The 1/5 success rule is a heuristic that suggests the mutation strength (sigma) should be 
    adjusted so that about 1/5 (20%) of mutations are successful. If more than 1/5 are successful,
    increase sigma to explore more; if fewer than 1/5 are successful, decrease sigma to refine the search.
    
    When the algorithm gets stuck in local optima, it performs restarts to new random positions.
    
    Supports optimization in different dimensions (1D, 2D, 30D, etc.).
    """
    
    def __init__(self, function, start_point=None, random_seed=None, sigma=0.5, objective=0, max_generations=10000, dimensions=2):
        #====== SETUP PARAMETERS: Define the target function and problem parameters ======#
        # Set random seed if provided, to ensure consistent start if start_point is None
        if random_seed is not None:
            seed(random_seed)               # Set the random seed for reproducible results
            np.random.seed(random_seed)     # Also set NumPy's random seed

        self.dimensions = dimensions        # Store the number of dimensions
        
        #====== INITIALIZE POPULATION: Create initial solution point ======#
        # In (1+1)-ES, the "population" is just a single individual
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

        self.start_point = self.position.copy()                     # Store initial point for reporting                  
        self.function = function                                    # The function to be optimized
        self.num_parameters = dimensions                            # Number of dimensions for optimization
        self.sigma = sigma                                          # Initial mutation step size
        self.objective = objective                                  # Target objective value (e.g., 0 for Rastrigin global min)
        self.max_error = 0.001                                      # Tolerance for stopping criterion. Lower values require more precise solutions
        self.max_generations = max_generations                      # Maximum number of generations to run. Higher values allow more time for convergence
        self.end_point = None                                       # Will store final solution
        self.initial_score = self.evaluate_position(self.position)  # Calculate score at starting point
        self.result = self.initial_score                            # Current best score
        self.error = abs(self.objective - self.result)              # Distance from target objective
        self.generations_run = 0                                    # Counter for generations actually run
        self.best_position = self.position.copy()                   # Track best solution found
        self.best_score = self.initial_score                        # Track best score

    def evaluate_position(self, position):
        """Evaluate the fitness of a position based on the number of dimensions"""
        if self.dimensions == 1:
            return self.function.evaluate_1d(position[0]) if hasattr(self.function, 'evaluate_1d') else self.function.evaluate(position)
        elif self.dimensions == 2:
            return self.function.evaluate_2d(position[0], position[1]) if hasattr(self.function, 'evaluate_2d') else self.function.evaluate(position)
        else:
            # For higher dimensions, use a general evaluate method if available
            return self.function.evaluate(position) if hasattr(self.function, 'evaluate') else sum([self.function.evaluate_1d(x) for x in position])

    def calculate_error(self):
        """Calculate the error as the absolute difference from the objective"""
        self.error = abs(self.objective - self.result)
        return self.error

    def evolve(self):
        """
        Run the optimization algorithm.
        
        The algorithm generates one offspring per generation using Gaussian mutation,
        keeps the better of parent and offspring, and adapts the mutation strength (sigma)
        according to the 1/5 success rule. It can restart from new random positions when
        it detects it's stuck in a local optimum.
        """
        #====== ITERATIVE EVOLUTION: Core evolution loop ======#
        count = 0           # Generation counter
        gen_sigma = 20      # Number of generations for calculating success rate
                              ## Higher values: more stable but slower adaptation
                              ## Lower values: faster adaptation but more noise
        successful_gen = 0  # Counter for successful mutations
        restarts = 0        # Counter for number of restarts performed
        max_restarts = 3    # Maximum number of restarts allowed
                              ## Higher values give more chances to escape local optima

        # Print initial status
        print(f"  Starting optimization with sigma: {self.sigma:.6f}, initial score: {self.result:.6f}")

        while count < self.max_generations and self.error > self.max_error:
            #====== GENERATE OFFSPRING & MUTATION: Create mutated solution ======#
            # Create a new offspring by applying mutations to each dimension
            offspring = self.position.copy()
            for i in range(self.dimensions):
                offspring[i] += gauss(0, self.sigma)  # Independent mutation for each dimension
            
            # Ensure we stay within bounds defined by the function object
            min_bound = self.function.min_bound if hasattr(self.function, 'min_bound') else -5.12
            max_bound = self.function.max_bound if hasattr(self.function, 'max_bound') else 5.12
            offspring = np.clip(offspring, min_bound, max_bound)
            
            #====== FITNESS EVALUATION: Calculate offspring fitness ======#
            # Evaluate the fitness of the offspring
            current_eval_score = self.evaluate_position(offspring)

            #====== SELECTION OF BEST INDIVIDUAL: Keep better solution ======#
            # Selection: If the offspring is better, it replaces the parent
            if current_eval_score < self.result:
                self.position = offspring.copy()                 # Update current solution
                self.result = current_eval_score                 # Update current best score
                successful_gen += 1                              # Increment successful mutation counter
                
                # Update overall best solution found so far (in case we restart)
                if current_eval_score < self.best_score:
                    self.best_score = current_eval_score
                    self.best_position = offspring.copy()

            self.calculate_error()  # Recalculate error based on current result
            
            #====== UPDATE GENERATION COUNT: Increment counter and check stopping conditions ======#
            count += 1              # Increment generation counter

            # 1/5th success rule implementation:
            # Every gen_sigma generations, check the success rate and adjust sigma
            if count % gen_sigma == 0 and gen_sigma > 0: 
                success_prob_window = successful_gen / gen_sigma    # Success rate over the window
                successful_gen = 0                                  # Reset counter for next window
                
                # Only log progress occasionally
                if count % 1000 == 0: 
                    print(f"  Generation {count}: Success probability: {success_prob_window:.4f}, sigma: {self.sigma:.6f}, best score: {self.best_score:.6f}")
                
                # Adjust sigma according to 1/5 rule:
                ## - If success rate > 0.2, decrease sigma (less exploration)
                ## - If success rate < 0.2, increase sigma (more exploration)
                if success_prob_window > 0.2:
                    self.sigma /= ADAPT_CONST  # Decrease sigma
                elif success_prob_window < 0.2:
                    self.sigma *= ADAPT_CONST  # Increase sigma
                
                # Prevent sigma from becoming too small or too large
                self.sigma = max(1e-7, min(self.sigma, 1.0))  # Keep sigma within sensible bounds

                # Restart mechanism:
                # If sigma is too small AND we haven't restarted too many times AND success rate is low
                # then restart from a new random position
                if self.sigma < 1e-6 and restarts < max_restarts and success_prob_window < 0.05: 
                    print(f"    -Restarting search (attempt {restarts+1}/{max_restarts}) at generation {count} due to low sigma and progress.")
                    
                    # Generate a new random position for restart
                    min_bound = self.function.min_bound if hasattr(self.function, 'min_bound') else -5.12
                    max_bound = self.function.max_bound if hasattr(self.function, 'max_bound') else 5.12
                    self.position = np.random.uniform(min_bound, max_bound, self.dimensions)
                    
                    self.sigma = 0.5                                         # Reset sigma to initial value
                    self.result = self.evaluate_position(self.position)      # Evaluate new point
                    successful_gen = 0                                       # Reset success counter
                    restarts += 1                                            # Increment restart counter

        # After search is complete, use the best solution found across all runs
        self.position = self.best_position.copy()
        self.result = self.best_score 
        self.generations_run = count
        self.end_point = self.best_position.copy()
        self.calculate_error()  # Final error calculation
        
        # Print final status
        print(f"  Optimization completed after {count} generations with final score: {self.result:.6f}")