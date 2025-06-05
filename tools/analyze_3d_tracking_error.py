"""
3D Tracking Error Analysis

This script analyzes the error between 3D tracked coordinates and the real grid positions.
It compares points from 3d_coordinates.json with the expected grid coordinates and
calculates error statistics.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import Normalize
from grid_generator import generate_grid

def load_3d_coordinates(json_path):
    """
    Load 3D coordinates data from a JSON file.
    
    Args:
        json_path (str): Path to the 3D coordinates JSON file
        
    Returns:
        list: List of dictionaries containing timestamp, capture_name, and points
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {json_path}")
        return data
    except Exception as e:
        print(f"Error loading 3D coordinates: {e}")
        return []

def generate_reference_grid(center_x=0, center_y=0, grid_size=7, spacing=0.5):
    """
    Generate a reference grid using the grid_generator functionality.
    
    Args:
        center_x, center_y: Center coordinates of the grid
        grid_size (int): Grid size (number of points in each dimension)
        spacing (float): Distance between adjacent grid points
        
    Returns:
        numpy.ndarray: Array of grid coordinates (n×2)
    """
    X, Y = generate_grid(center_x, center_y, grid_size, spacing)
    grid_points = []
    
    # Flatten the grid coordinates into a list of points
    for i in range(grid_size):
        for j in range(grid_size):
            grid_points.append([X[i, j], Y[i, j], 0])  # Assume Z=0 for grid points

    new_points = []
    for point in grid_points:
        if point == [1,-1.5,0] or point == [1.5,-1.5,0]:
            # Skip these specific points as per the original code logic
            continue
        else:
            new_points.append(point)


    return np.array(new_points)

def find_nearest_grid_point(tracked_point, grid_points):
    """
    Find the nearest grid point to a tracked 3D point.
    
    Args:
        tracked_point (numpy.ndarray): The tracked 3D point
        grid_points (numpy.ndarray): Array of grid points
        
    Returns:
        tuple: (nearest grid point, distance to nearest grid point, index of nearest point)
    """
    # Calculate Euclidean distance in the XY plane only (ignore Z)
    distances = np.sqrt(
        np.sum((grid_points[:, :2] - tracked_point[:2]) ** 2, axis=1)
    )
    nearest_idx = np.argmin(distances)
    
    return grid_points[nearest_idx], distances[nearest_idx], nearest_idx

def analyze_tracking_error(tracked_points_data, grid_points, max_distance=0.5):
    """
    Analyze errors between tracked points and reference grid.
    
    Args:
        tracked_points_data (list): List of tracked points data from JSON
        grid_points (numpy.ndarray): Array of reference grid points
        max_distance (float): Maximum distance for a valid match
        
    Returns:
        tuple: (matched_pairs, errors, error_stats)
    """
    matched_pairs = []
    errors = []
    unmatched_points = []
    
    # Extract all tracked points
    all_tracked_points = []
    capture_indices = []
    
    for capture_idx, capture in enumerate(tracked_points_data):
        for point in capture['points']:
            all_tracked_points.append(point)
            capture_indices.append(capture_idx)
    
    # Convert to numpy array for easier processing
    if all_tracked_points:
        all_tracked_points = np.array(all_tracked_points)
        
        # Find nearest grid point for each tracked point
        for i, point in enumerate(all_tracked_points):
            nearest_grid_point, distance, grid_idx = find_nearest_grid_point(point, grid_points)
            
            # Only consider points within max_distance as valid matches
            if distance <= max_distance:
                # Calculate coordinate-wise errors
                error_x = point[0] - nearest_grid_point[0]
                error_y = point[1] - nearest_grid_point[1]
                error_z = point[2] - nearest_grid_point[2]
                
                matched_pairs.append({
                    'tracked_point': point,
                    'grid_point': nearest_grid_point,
                    'distance': distance,
                    'error_x': error_x,
                    'error_y': error_y,
                    'error_z': error_z,
                    'capture_name': tracked_points_data[capture_indices[i]]['capture_name']
                })
                errors.append(distance)
            else:
                unmatched_points.append({
                    'tracked_point': point,
                    'nearest_distance': distance,
                    'capture_name': tracked_points_data[capture_indices[i]]['capture_name']
                })
    
    # Calculate error statistics
    if errors:
        # Overall distance errors
        error_stats = {
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'num_matched': len(errors),
            'num_unmatched': len(unmatched_points),
            'matching_rate': len(errors) / (len(errors) + len(unmatched_points)) * 100
        }
        
        # Coordinate-wise errors
        errors_x = [pair['error_x'] for pair in matched_pairs]
        errors_y = [pair['error_y'] for pair in matched_pairs]
        errors_z = [pair['error_z'] for pair in matched_pairs]
        
        error_stats.update({
            'mean_error_x': np.mean(errors_x),
            'mean_error_y': np.mean(errors_y),
            'mean_error_z': np.mean(errors_z),
            'std_error_x': np.std(errors_x),
            'std_error_y': np.std(errors_y),
            'std_error_z': np.std(errors_z),
            'rmse_x': np.sqrt(np.mean(np.array(errors_x)**2)),
            'rmse_y': np.sqrt(np.mean(np.array(errors_y)**2)),
            'rmse_z': np.sqrt(np.mean(np.array(errors_z)**2))
        })
    else:
        error_stats = {
            'min_error': 0,
            'max_error': 0,
            'mean_error': 0,
            'median_error': 0,
            'std_error': 0,
            'num_matched': 0,
            'num_unmatched': len(unmatched_points),
            'matching_rate': 0,
            'mean_error_x': 0,
            'mean_error_y': 0,
            'mean_error_z': 0,
            'std_error_x': 0,
            'std_error_y': 0,
            'std_error_z': 0,
            'rmse_x': 0,
            'rmse_y': 0,
            'rmse_z': 0
        }
    
    return matched_pairs, errors, error_stats, unmatched_points

def visualize_tracking_error(matched_pairs, unmatched_points, grid_points, error_stats):
    """
    Visualize tracking errors between matched points and the grid.
    
    Args:
        matched_pairs (list): List of matched pairs (tracked point and grid point)
        unmatched_points (list): List of unmatched tracked points
        grid_points (numpy.ndarray): Array of grid points
        error_stats (dict): Error statistics
    """
    # Create figure with 2D and 3D plots stacked vertically
    fig = plt.figure(figsize=(12, 16))
    
    # 2D Plot (top)
    ax_2d = fig.add_subplot(211)
    
    # 3D Plot (bottom)
    ax_3d = fig.add_subplot(212, projection='3d')
    
    # Plot grid points
    ax_2d.scatter(grid_points[:, 0], grid_points[:, 1], color='blue', marker='o', s=50)
    ax_3d.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], color='blue', marker='o', s=50, label='Grid Points')
    
    # Add grid lines in 2D view
    x_vals = sorted(set(grid_points[:, 0]))
    y_vals = sorted(set(grid_points[:, 1]))
    
    for x in x_vals:
        ax_2d.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
    
    for y in y_vals:
        ax_2d.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
    
    # Create a colormap for errors
    if matched_pairs:
        distances = [pair['distance'] for pair in matched_pairs]
        norm = Normalize(vmin=0, vmax=max(distances))
        
        # Extract points for plotting
        tracked_points = np.array([pair['tracked_point'] for pair in matched_pairs])
        grid_matched = np.array([pair['grid_point'] for pair in matched_pairs])
        
        # Plot matched tracked points with error color
        scatter_2d = ax_2d.scatter(
            tracked_points[:, 0], tracked_points[:, 1], 
            c=distances, cmap='viridis_r', s=80, alpha=0.8,
            marker='x'
        )
        
        scatter_3d = ax_3d.scatter(
            tracked_points[:, 0], tracked_points[:, 1], tracked_points[:, 2], 
            c=distances, cmap='viridis_r', s=80, alpha=0.8,
            marker='x', label='Tracked Points (matched)'
        )
        
        # Connect matched points with lines
        for i, pair in enumerate(matched_pairs):
            tracked = pair['tracked_point']
            grid = pair['grid_point']
            
            # 2D plot
            ax_2d.plot([tracked[0], grid[0]], [tracked[1], grid[1]], 'r-', alpha=0.3)
            
            # 3D plot
            ax_3d.plot([tracked[0], grid[0]], [tracked[1], grid[1]], [tracked[2], grid[2]], 'r-', alpha=0.3)
        
        # Add colorbar
        cbar = fig.colorbar(scatter_3d, ax=ax_3d, pad=0.1)
        cbar.set_label('Error (meters)')
    
    # Plot unmatched points
    if unmatched_points:
        unmatched_array = np.array([p['tracked_point'] for p in unmatched_points])
        ax_2d.scatter(
            unmatched_array[:, 0], unmatched_array[:, 1],
            color='red', marker='x', s=80, alpha=0.5,
            label=f'Unmatched Points ({len(unmatched_points)})'
        )
        
        ax_3d.scatter(
            unmatched_array[:, 0], unmatched_array[:, 1], unmatched_array[:, 2],
            color='red', marker='x', s=80, alpha=0.5,
            label=f'Unmatched Points ({len(unmatched_points)})'
        )
    
    # Set labels and titles
    ax_2d.set_xlabel('X (meters)')
    ax_2d.set_ylabel('Y (meters)')
    ax_2d.set_title('Top-Down View of Tracking Errors')
    
    ax_3d.set_xlabel('X (meters)')
    ax_3d.set_ylabel('Y (meters)')
    ax_3d.set_zlabel('Z (meters)')
    ax_3d.set_title('3D View of Tracking Errors')
    
    # Set fixed Z-axis limits between -1 and 1
    ax_3d.set_zlim(-1, 1)
    
    # Add statistics to figure
    stats_text = (
        f"Error Statistics:\n"
        f"Mean: {error_stats['mean_error']:.3f}m\n"
        f"Std Dev: {error_stats['std_error']:.3f}m\n"
        f"Matched Points: {error_stats['num_matched']}\n"
        f"Unmatched Points: {error_stats['num_unmatched']}\n"
        f"Matching Rate: {error_stats['matching_rate']:.1f}%"
    )
    
    
    # Set equal aspect ratio for 2D plot
    ax_2d.axis('equal')
    
    # Add legends
    ax_2d.legend()
    ax_3d.legend()
    
    # Set main title
    plt.suptitle('3D Tracking Error Analysis', fontsize=16)
    
    # Tight layout with more space for vertical arrangement
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Show plot
    plt.show()

def generate_error_report(matched_pairs, error_stats, output_file=None):
    """
    Generate a detailed error report and save to file.
    
    Args:
        matched_pairs (list): List of matched pairs (tracked point and grid point)
        error_stats (dict): Error statistics
        output_file (str, optional): Path to output report file
    """
    report = []
    report.append("=" * 80)
    report.append("3D TRACKING ERROR ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Add overall error statistics
    report.append("OVERALL ERROR STATISTICS:")
    report.append(f"  Min Error: {error_stats['min_error']:.3f} meters")
    report.append(f"  Max Error: {error_stats['max_error']:.3f} meters")
    report.append(f"  Mean Error: {error_stats['mean_error']:.3f} meters")
    report.append(f"  Median Error: {error_stats['median_error']:.3f} meters")
    report.append(f"  Standard Deviation: {error_stats['std_error']:.3f} meters")
    report.append(f"  Number of Matched Points: {error_stats['num_matched']}")
    report.append(f"  Number of Unmatched Points: {error_stats['num_unmatched']}")
    report.append(f"  Matching Rate: {error_stats['matching_rate']:.1f}%")
    report.append("")
    
    # Add coordinate-wise error statistics
    report.append("COORDINATE-WISE ERROR STATISTICS:")
    report.append(f"  X-axis - Mean: {error_stats['mean_error_x']:.3f}m, Std: {error_stats['std_error_x']:.3f}m, RMSE: {error_stats['rmse_x']:.3f}m")
    report.append(f"  Y-axis - Mean: {error_stats['mean_error_y']:.3f}m, Std: {error_stats['std_error_y']:.3f}m, RMSE: {error_stats['rmse_y']:.3f}m")
    report.append(f"  Z-axis - Mean: {error_stats['mean_error_z']:.3f}m, Std: {error_stats['std_error_z']:.3f}m, RMSE: {error_stats['rmse_z']:.3f}m")
    report.append("")
    
    # Add detailed error information for each matched point
    report.append("DETAILED ERRORS BY CAPTURE:")
    report.append("-" * 80)
    report.append(f"{'Capture Name':<25} | {'Total(m)':<8} | {'X(m)':<7} | {'Y(m)':<7} | {'Z(m)':<7} | {'Grid Point':<15} | {'Tracked Point':<15}")
    report.append("-" * 80)
    
    # Group by capture name
    captures = {}
    for pair in matched_pairs:
        capture_name = pair['capture_name']
        if capture_name not in captures:
            captures[capture_name] = []
        captures[capture_name].append(pair)
    
    # Print errors for each capture
    for capture_name, pairs in sorted(captures.items()):
        for pair in pairs:
            grid_str = f"({pair['grid_point'][0]:.2f},{pair['grid_point'][1]:.2f})"
            tracked_str = f"({pair['tracked_point'][0]:.2f},{pair['tracked_point'][1]:.2f})"
            report.append(
                f"{capture_name:<25} | {pair['distance']:<8.3f} | "
                f"{pair['error_x']:<+7.3f} | {pair['error_y']:<+7.3f} | {pair['error_z']:<+7.3f} | "
                f"{grid_str:<15} | {tracked_str:<15}"
            )
    
    # Print report to console
    print("\n".join(report))
    
    # Save report to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("\n".join(report))
        print(f"Error report saved to {output_file}")

def main():
    """Main function to analyze 3D tracking errors."""
    parser = argparse.ArgumentParser(description="Analyze errors between tracked 3D points and reference grid")
    parser.add_argument("--coordinates", default="experiments/ground_truth/people_standing_grid_3d_coordinates.json",
                        help="Path to JSON file with 3D coordinates")
    parser.add_argument("--center-x", type=float, default=0,
                        help="X-coordinate of the grid center (default: 0)")
    parser.add_argument("--center-y", type=float, default=0,
                        help="Y-coordinate of the grid center (default: 0)")
    parser.add_argument("--grid-size", type=int, default=7,
                        help="Grid size (number of points in each dimension, default: 7)")
    parser.add_argument("--spacing", type=float, default=0.5,
                        help="Spacing between grid points in meters (default: 0.5)")
    parser.add_argument("--max-distance", type=float, default=0.5,
                        help="Maximum distance for a point to be considered a match (default: 0.5)")
    parser.add_argument("--report", default="tracking_error_report2.txt",
                        help="Path to output error report file")
    
    args = parser.parse_args()
    
    # Load tracked points
    tracked_points_data = load_3d_coordinates(args.coordinates)
    if not tracked_points_data:
        print("No tracked points data found. Exiting.")
        return
    
    # Generate reference grid
    print(f"Generating reference grid (center: {args.center_x}, {args.center_y}, size: {args.grid_size}, spacing: {args.spacing})")
    grid_points = generate_reference_grid(
        args.center_x, args.center_y, args.grid_size, args.spacing
    )
    
    # Analyze errors
    print(f"Analyzing tracking errors (max matching distance: {args.max_distance}m)")
    matched_pairs, errors, error_stats, unmatched_points = analyze_tracking_error(
        tracked_points_data, grid_points, args.max_distance
    )
    
    # Generate report
    generate_error_report(matched_pairs, error_stats, args.report)
    
    # Visualize errors
    visualize_tracking_error(matched_pairs, unmatched_points, grid_points, error_stats)

if __name__ == "__main__":
    main()
