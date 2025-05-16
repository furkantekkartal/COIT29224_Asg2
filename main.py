"""
@author: 12223508

Main script for running and visualizing optimization algorithms on the Rastrigin function.
"""

import os                       # For output file operations
import sys                      # For system operations   
from random import seed         # For random number generation
import time                     # For measuring execution time
import numpy as np              # For array operations
from tabulate import tabulate   # For table formatting

# Add the Code directory to the Python path for modular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(current_dir, "Code")
sys.path.append(code_dir)

# Import optimization classes
from RastriginFunction import RastriginFunction
from OnePlusOneESFifthRule import OnePlusOneESFifthRule
from GradientDescent import GradientDescent 
from Visualizer import Visualizer 

# --- Helper function to run and visualize optimization --- #
def run_and_visualize_optimization(
    algorithm_class,
    function_to_optimize,
    algorithm_name="DefaultAlgorithmName",
    random_seed_value=42,
    max_generations=10000,
    sigma=0.5,
    learning_rate=0.01,  # Learning rate for Gradient Descent
    start_point_coords=None, 
    save_plot_to_output=True,
    dimensions=2
):
    """
    Runs an optimization algorithm on a given function and visualizes the results.

    Args:
        algorithm_class: The class of the optimization algorithm to use.
        function_to_optimize: An instance of the function to be optimized.
        algorithm_name (str): Name of the algorithm for titles and filenames.
        random_seed_value (int): Seed for random number generation.
        max_generations (int): Maximum number of generations for the algorithm.
        sigma (float): Initial sigma value for Evolution Strategies.
        learning_rate (float): Learning rate for gradient-based algorithms.
        start_point_coords (tuple, optional): Specific starting point. If None, uses default.
        save_plot_to_output (bool): Whether to save the plot to the Output directory.
        dimensions (int): Number of dimensions for the optimization problem.
    """
    # Print Parameters
    print("\nParameters;  ")
    print(f"  Dimensions: {dimensions}D ")
    print(f"  Max Generations: {max_generations}")
    if algorithm_class == OnePlusOneESFifthRule:
        print(f"  Initial Sigma: {sigma}")
    elif algorithm_class == GradientDescent:
        print(f"  Initial Learning Rate: {learning_rate}")
    print(f"  Random Seed: {random_seed_value}")
    
    if start_point_coords:
        print(f"  Custom Start Point: {start_point_coords}")

    # Create output directory if it doesn't exist
    output_dir = os.path.join(current_dir, "Output")
    if save_plot_to_output and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Visualize the function being optimized
    visualizer = Visualizer(function_to_optimize)

    # Start timing
    start_time = time.time()

    # --- Operation ---
    print("\nOperation;  ")
    # Instantiate the optimizer based on the algorithm class
    if algorithm_class == OnePlusOneESFifthRule:
        optimizer = algorithm_class(
            function=function_to_optimize,
            start_point=start_point_coords,
            random_seed=random_seed_value,
            sigma=sigma,
            objective=function_to_optimize.global_minimum_value,
            max_generations=max_generations,
            dimensions=dimensions
        )
    elif algorithm_class == GradientDescent:
        optimizer = algorithm_class(
            function=function_to_optimize,
            start_point=start_point_coords,
            random_seed=random_seed_value,
            learning_rate=learning_rate,
            objective=function_to_optimize.global_minimum_value,
            max_generations=max_generations,
            dimensions=dimensions
        )
    else:
        raise ValueError(f"Unsupported algorithm class: {algorithm_class.__name__}")

    # Run the optimization
    optimizer.evolve()

    # End timing
    execution_time = time.time() - start_time

    # --- Results ---
    print("\nResults;")
    print(f"  Generations Run   : {optimizer.generations_run}")
    print(f"  Initial Score     : {optimizer.initial_score:.2f}")
    print(f"  Final Score       : {optimizer.result:.2f}")
    print(f"  Execution Time    : {execution_time:.2f} seconds")

    # Calculate mean squared error if global minimum is known
    mse = None
    if hasattr(function_to_optimize, 'global_minimum'):
        if dimensions == 1:
            # For 1D, compare with first coordinate of global minimum
            mse = (optimizer.end_point[0] - function_to_optimize.global_minimum[0])**2
        elif dimensions == 2:
            # For 2D, compare with both coordinates
            mse = ((optimizer.end_point[0] - function_to_optimize.global_minimum[0])**2 + 
                   (optimizer.end_point[1] - function_to_optimize.global_minimum[1])**2) / 2
        else:
            # For higher dimensions, if global_minimum is defined for that many dimensions
            if hasattr(function_to_optimize, 'global_minimum_nd') and len(function_to_optimize.global_minimum_nd) == dimensions:
                squared_errors = [(optimizer.end_point[i] - function_to_optimize.global_minimum_nd[i])**2 for i in range(dimensions)]
                mse = sum(squared_errors) / dimensions
    
    if mse is not None:
        print(f"  Mean Squared Error: {mse:.6f}")

    # --- Output ---
    print("\nOutput;")
    # Visualize results based on dimensions
    if save_plot_to_output:
        plot_filename = os.path.join(output_dir, f"{algorithm_name}_{dimensions}D_on_{function_to_optimize.__class__.__name__}.png")
        if dimensions == 1:
            # 1D visualization
            visualizer.visualize_1d(
                algorithm_name=f"{algorithm_name} ({dimensions}D)",
                start_point=optimizer.start_point,
                end_point=optimizer.end_point,
                save_filename=plot_filename
            )
            print(f"  1D plot saved: {os.path.basename(plot_filename)}")
        elif dimensions == 2:
            # 2D visualization (standard)
            visualizer.visualize_2d(
                algorithm_name=f"{algorithm_name} ({dimensions}D)",
                start_point=optimizer.start_point,
                end_point=optimizer.end_point,
                save_filename=plot_filename
            )
            print(f"  Plot saved: {os.path.basename(plot_filename)}")
        else:
            # High-dimensional visualization
            visualizer.visualize_high_d(
                algorithm_name=f"{algorithm_name} ({dimensions}D)",
                start_point=optimizer.start_point,
                end_point=optimizer.end_point,
                save_filename=plot_filename
            )
            print(f"  Plot saved: {os.path.basename(plot_filename)}")
    
    print("\n") # Add a newline for spacing before the next run

    # Return metrics for comparison
    return {
        "algorithm": algorithm_name,
        "dimensions": dimensions,
        "generations": optimizer.generations_run,
        "initial_score": optimizer.initial_score,
        "final_score": optimizer.result,
        "execution_time": execution_time,
        "error": optimizer.error,
        "mse": mse if mse is not None else "N/A"
    }

# --- Main execution block --- #
def main():
    """Main function to set up and run optimizations."""
    
    # Set global seed for reproducibility
    seed(26) 
    np.random.seed(26)  # Seed numpy

    # Instantiate the function to be optimized
    rastrigin = RastriginFunction()

    # Lists to store results for comparison
    es_results = []
    gd_results = []

    # Run the algorithm for different dimensions
    dimensions_to_test = [1, 2, 30]
    
    # First, run the Evolution Strategy for all dimensions
    print("\n========== RUNNING EVOLUTION STRATEGY OPTIMIZATION ==========")
    for dim in dimensions_to_test:
        # Let algorithm initialize using its random_seed logic
        specific_start_point = None
        
        print(f"\n=== Running OnePlusOneESFifthRule on {dim}D Rastrigin ===")
        result = run_and_visualize_optimization(
            algorithm_class=OnePlusOneESFifthRule,
            function_to_optimize=rastrigin,
            algorithm_name="OnePlusOneESFifthRule",
            random_seed_value=42,
            max_generations=10000, 
            sigma=0.5,            
            start_point_coords=specific_start_point,
            save_plot_to_output=True,
            dimensions=dim
        )
        
        es_results.append(result)

    # Next, run Gradient Descent for all dimensions
    print("\n========== RUNNING GRADIENT DESCENT OPTIMIZATION ==========")
    for dim in dimensions_to_test:
        # Use the same starting point seed for fair comparison
        specific_start_point = None
        
        print(f"\n=== Running GradientDescent on {dim}D Rastrigin ===")
        result = run_and_visualize_optimization(
            algorithm_class=GradientDescent,
            function_to_optimize=rastrigin,
            algorithm_name="GradientDescent",
            random_seed_value=42,
            max_generations=10000, 
            learning_rate=0.01,
            start_point_coords=specific_start_point,
            save_plot_to_output=True,
            dimensions=dim
        )
        
        gd_results.append(result)

    # Compare ES vs GD for each dimension
    print("\n========== EVOLUTION STRATEGY vs GRADIENT DESCENT COMPARISON ==========")
    
    for i, dim in enumerate(dimensions_to_test):
        print(f"\n--- Dimension: {dim}D ---")
        
        headers_comp = ["Metric", "Evolution Strategy", "Baseline Opt. Alg."]
        table_data_comp = [
            ["Algorithm", es_results[i]["algorithm"], gd_results[i]["algorithm"]],
            ["Generations", es_results[i]["generations"], gd_results[i]["generations"]],
            ["Initial Score", f"{es_results[i]['initial_score']:.2f}", f"{gd_results[i]['initial_score']:.2f}"],
            ["Final Score", f"{es_results[i]['final_score']:.2f}", f"{gd_results[i]['final_score']:.2f}"],
            ["MSE from Global Min", f"{es_results[i]['mse']:.6f}" if isinstance(es_results[i]['mse'], float) else es_results[i]['mse'], 
                                  f"{gd_results[i]['mse']:.6f}" if isinstance(gd_results[i]['mse'], float) else gd_results[i]['mse']],
            ["Execution Time (s)", f"{es_results[i]['execution_time']:.2f}", f"{gd_results[i]['execution_time']:.2f}"]
        ]
        
        print(tabulate(table_data_comp, headers=headers_comp, tablefmt="grid"))
    
    # Summary analysis
    print("\n========== PERFORMANCE SUMMARY ==========")
    
    # Calculate score reduction percentages
    es_reductions = [(result["initial_score"] - result["final_score"]) / result["initial_score"] * 100 for result in es_results]
    gd_reductions = [(result["initial_score"] - result["final_score"]) / result["initial_score"] * 100 for result in gd_results]
    
    # Calculate efficiency (score reduction per second)
    es_efficiency = [(result["initial_score"] - result["final_score"]) / result["execution_time"] if result["execution_time"] > 0 else 0 for result in es_results]
    gd_efficiency = [(result["initial_score"] - result["final_score"]) / result["execution_time"] if result["execution_time"] > 0 else 0 for result in gd_results]
    
    # Create summary table
    headers_summary = ["Metric", "1D", "2D", "30D"]
    table_data_summary = [
        ["ES Score Reduction (%)", f"{es_reductions[0]:.2f}%", f"{es_reductions[1]:.2f}%", f"{es_reductions[2]:.2f}%"],
        ["GD Score Reduction (%)", f"{gd_reductions[0]:.2f}%", f"{gd_reductions[1]:.2f}%", f"{gd_reductions[2]:.2f}%"],
        ["ES Efficiency (score/s)", f"{es_efficiency[0]:.2f}", f"{es_efficiency[1]:.2f}", f"{es_efficiency[2]:.2f}"],
        ["GD Efficiency (score/s)", f"{gd_efficiency[0]:.2f}", f"{gd_efficiency[1]:.2f}", f"{gd_efficiency[2]:.2f}"]
    ]
    
    print(tabulate(table_data_summary, headers=headers_summary, tablefmt="grid"))
    
    # Visualize performance metrics for each algorithm
    visualizer = Visualizer(rastrigin)
    output_dir = os.path.join(current_dir, "Output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Visualize ES performance metrics
    es_metrics_filename = os.path.join(output_dir, "ES_Performance_Metrics_Comparison.png")
    visualizer.visualize_performance_metrics(es_results, "OnePlusOneESFifthRule", save_filename=es_metrics_filename)
    
    # Visualize GD performance metrics
    gd_metrics_filename = os.path.join(output_dir, "GD_Performance_Metrics_Comparison.png")
    visualizer.visualize_performance_metrics(gd_results, "GradientDescent", save_filename=gd_metrics_filename)

    print("\nOutput:") # Match the example
    print(f"  Plot saved: {os.path.basename(es_metrics_filename)}")
    print(f"  Plot saved: {os.path.basename(gd_metrics_filename)}")

    print("\n======================== THE END ========================")

if __name__ == '__main__':
    main()