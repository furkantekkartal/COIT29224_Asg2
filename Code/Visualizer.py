import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    """Class for visualizing optimization results on benchmark functions"""
    
    def __init__(self, function):
        self.function = function
        
    def visualize_1d(self, algorithm_name, start_point, end_point, save_filename=None):
        """
        Visualize the optimization path on a 1D function.
        """
        # Create x values for plotting
        x_vals = np.linspace(self.function.min_bound, self.function.max_bound, 1000)
        
        # Calculate function values
        y_vals = np.array([self.function.evaluate_1d(x) for x in x_vals])
        
        # Create the figure
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        # Plot the function
        ax.plot(x_vals, y_vals, 'b-', label='Function')
        
        # Evaluate and plot start and end points
        start_y = self.function.evaluate_1d(start_point[0])
        end_y = self.function.evaluate_1d(end_point[0])
        ax.plot(start_point[0], start_y, 'ro', markersize=10, label='Start Point')
        ax.plot(end_point[0], end_y, 'go', markersize=10, label='End Point')
        
        # Plot global minimum if available
        if hasattr(self.function, 'global_minimum'):
            global_min_x = self.function.global_minimum[0]
            ax.plot(global_min_x, self.function.global_minimum_value, 'y*', markersize=15, label='Global Minimum')
        
        # Add labels and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Function Value')
        ax.set_title(f'{algorithm_name} on {self.function.__class__.__name__} (1D)')
        ax.legend()
        ax.grid(True)
        
        # Save plot if filename is provided
        if save_filename:
            plt.savefig(save_filename)

    def visualize_2d(self, algorithm_name, start_point, end_point, save_filename=None):
        """
        Visualize the optimization path on a 2D function.
        """
        # Create meshgrid for plotting
        x_vals = np.linspace(self.function.min_bound, self.function.max_bound, 400)
        y_vals = np.linspace(self.function.min_bound, self.function.max_bound, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = self.function.evaluate_2d(X, Y)

        fig = plt.figure(figsize=(14, 7))

        # 2D Contour Plot
        ax2 = fig.add_subplot(121)
        contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
        fig.colorbar(contour, ax=ax2)
        ax2.set_title(f'{algorithm_name} on {self.function.__class__.__name__} (2D Contour)')
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')
        ax2.plot(start_point[0], start_point[1], 'ro', label='Start Point')
        ax2.plot(end_point[0], end_point[1], 'go', label='End Point')
        if hasattr(self.function, 'global_minimum'):
            ax2.plot(self.function.global_minimum[0], self.function.global_minimum[1], 
                     'y*', markersize=15, label='Global Minimum')
        ax2.legend()
        ax2.set_xlim(self.function.min_bound, self.function.max_bound)
        ax2.set_ylim(self.function.min_bound, self.function.max_bound)

        # 3D Surface Plot
        ax1 = fig.add_subplot(122, projection='3d')
        ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax1.set_title(f'{algorithm_name} on {self.function.__class__.__name__} (3D)')
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Y-axis')
        ax1.set_zlabel('Z-axis')
        
        start_z = self.function.evaluate_2d(start_point[0], start_point[1])
        end_z = self.function.evaluate_2d(end_point[0], end_point[1])
        
        ax1.scatter(start_point[0], start_point[1], start_z, 
                    color='red', s=100, label='Start Point', depthshade=True)
        ax1.scatter(end_point[0], end_point[1], end_z, 
                    color='green', s=100, label='End Point', depthshade=True)
        if hasattr(self.function, 'global_minimum_value'):
            ax1.scatter(self.function.global_minimum[0], self.function.global_minimum[1], self.function.global_minimum_value, 
                        color='yellow', s=200, marker='*', label='Global Minimum', depthshade=True)
        ax1.legend()
        ax1.set_xlim(self.function.min_bound, self.function.max_bound)
        ax1.set_ylim(self.function.min_bound, self.function.max_bound)

        plt.tight_layout()
        
        if save_filename:
            plt.savefig(save_filename)

    def visualize_high_d(self, algorithm_name, start_point, end_point, save_filename=None):
        """
        Visualize high-dimensional optimization results using a radar chart.
        """
        dimensions = len(start_point)
        
        # Create radar chart
        fig, ax1 = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Set up angles for radar chart
        angles = np.linspace(0, 2 * np.pi, dimensions, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Use the actual values directly without normalization
        min_bound = self.function.min_bound if hasattr(self.function, 'min_bound') else -5.12
        max_bound = self.function.max_bound if hasattr(self.function, 'max_bound') else 5.12
        
        # Prepare data for radar chart - use actual values
        start_values = list(start_point) + [start_point[0]]  # Close the loop
        end_values = list(end_point) + [end_point[0]]  # Close the loop
        
        # Create global minimum circle (all zeros for Rastrigin)
        global_min_values = [0] * dimensions + [0]  # Close the loop
        
        # Set up the radar chart limits and grid
        ax1.set_ylim(min_bound, max_bound)
        
        # Add circular grid lines
        grid_values = np.linspace(min_bound, max_bound, 0)
        
        # Plot radar chart
        for i in range(dimensions):
            # Get the current angle for this dimension
            angle = angles[i]
            
            # Plot start and end points as markers
            ax1.plot([angle], [start_values[i]], 'ro', markersize=8)  # Start point (red dot)
            ax1.plot([angle], [end_values[i]], 'go', markersize=8)    # End point (green dot)
            
            # Draw a line connecting start and end points for this dimension
            ax1.plot([angle, angle], [start_values[i], end_values[i]], '-', color='black', alpha=1)

        # Add labels at the 12 o'clock position
        ax1.text(1.57, 5.12, '5.12', horizontalalignment='center', verticalalignment='top', fontsize=15)
        ax1.text(1.57, 0, '0', horizontalalignment='center', verticalalignment='top', fontsize=15)
        ax1.text(1.57, -5.12, '-5.12', horizontalalignment='center', verticalalignment='bottom', fontsize=15)

        ax1.plot(angles, global_min_values, 'y-', linewidth=2, label='Global Minimum')
        ax1.fill(angles, global_min_values, 'y', alpha=0.2)

        # Add the legend after all plotting commands
        ax1.plot([angle], [start_values[1]], 'ro', markersize=6, label='Start Point')  # Start point (red dot)
        ax1.plot([angle], [end_values[1]], 'go', markersize=6, label='End Point')    # End point (green dot)
        ax1.legend(loc='upper right')
        
        # Set labels
        labels = [f'D{i+1}' for i in range(dimensions)]
        labels += [labels[0]]  # Close the loop
        
        # Display spaced labels
        num_labels = min(dimensions, 30) 
        if dimensions > num_labels:
            step = dimensions // num_labels
            selected_angles = angles[::step]
            selected_labels = [f'D{i+1}' for i in range(0, dimensions, step)]
            # Make sure we have the same number of labels as angles
            if len(selected_angles) > len(selected_labels):
                selected_angles = selected_angles[:len(selected_labels)]
            ax1.set_xticks(selected_angles)
            ax1.set_xticklabels(selected_labels)
        else:
            ax1.set_xticks(angles)
            ax1.set_xticklabels(labels)
        
        # Add radial axis labels (showing the values)
        ax1.set_yticks(grid_values)
        ax1.set_yticklabels([f"{val:.1f}" for val in grid_values])
        
        ax1.set_title(f'{algorithm_name} on {self.function.__class__.__name__}\nParameter Values (Range: {min_bound} to {max_bound})')
        
        # Save if filename is provided
        if save_filename:
            plt.savefig(save_filename)

    def visualize_performance_metrics(self, results, algorithm_name, save_filename=None):
        """
        Visualize performance metrics using bar charts.
        """
        dimensions = [result["dimensions"] for result in results]
        final_scores = [result["final_score"] for result in results]
        execution_times = [result["execution_time"] for result in results]
        
        # Calculate local minima counts for each dimension
        local_minima_counts = [self.function.calculate_local_minima_count(dim) for dim in dimensions]
        
        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Convert dimensions to strings for bar chart labels
        dimension_labels = [f"{dim}D" for dim in dimensions]
        x_positions = np.arange(len(dimensions))
        
        # Plot final scores as bar chart
        ax1.bar(x_positions, final_scores, color='blue', alpha=0.7)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(dimension_labels)
        ax1.set_xlabel('Dimensions')
        ax1.set_ylabel('Final Score')
        ax1.set_title('Final Score by Dimension')
        for i, score in enumerate(final_scores):
            ax1.text(i, score, f'{score:.2f}', ha='center', va='bottom')
        ax1.grid(True, axis='y')
        
        # Plot execution times as bar chart
        ax2.bar(x_positions, execution_times, color='red', alpha=0.7)
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(dimension_labels)
        ax2.set_xlabel('Dimensions')
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_title('Execution Time by Dimension')
        for i, time in enumerate(execution_times):
            ax2.text(i, time, f'{time:.2f}s', ha='center', va='bottom')
        ax2.grid(True, axis='y')
        
        # Plot local minima counts as bar chart with logarithmic scale
        ax3.bar(x_positions, local_minima_counts, color='green', alpha=0.7)
        ax3.set_xticks(x_positions)
        ax3.set_xticklabels(dimension_labels)
        ax3.set_xlabel('Dimensions')
        ax3.set_ylabel('Local Minima Count (Log Scale)')
        ax3.set_title('Local Minima Count by Dimension')
        ax3.set_yscale('log')  # Set y-axis to logarithmic scale
        for i, count in enumerate(local_minima_counts):
            ax3.text(i, count, f'{count:,}', ha='center', va='bottom')  # Use comma for thousands separator
        ax3.grid(True, axis='y')
        
        # Plot score reduction percentages as bar chart
        reduction_percentages = [(result["initial_score"] - result["final_score"]) / result["initial_score"] * 100 for result in results]
        ax4.bar(x_positions, reduction_percentages, color='purple', alpha=0.7)
        ax4.set_xticks(x_positions)
        ax4.set_xticklabels(dimension_labels)
        ax4.set_xlabel('Dimensions')
        ax4.set_ylabel('Score Reduction (%)')
        ax4.set_title('Score Reduction Percentage by Dimension')
        for i, pct in enumerate(reduction_percentages):
            ax4.text(i, pct, f'{pct:.1f}%', ha='center', va='bottom')
        ax4.grid(True, axis='y')
        
        plt.suptitle(f'Performance Metrics Comparison for {algorithm_name} by Dimension', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save if filename is provided
        if save_filename:
            plt.savefig(save_filename)