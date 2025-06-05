"""
Reconstruction Error Data Adjustment Script

This script adjusts a reconstruction error analysis JSON file by:
1. Subtracting specified offsets from X and Y coordinates of both actual and estimated positions
2. Recalculating the error values based on the new positions
3. Updating error statistics
4. Saving to a new JSON file
"""

import json
import os
import math
import argparse
import numpy as np
from datetime import datetime

def load_reconstruction_data(input_file):
    """
    Load reconstruction error data from a JSON file.
    
    Args:
        input_file (str): Path to the input JSON file
        
    Returns:
        dict: Loaded reconstruction error data
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {input_file}")
        print(f"Found {data.get('total_captures', 0)} position comparisons")
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def calculate_error(actual, estimated):
    """
    Calculate the Euclidean distance between actual and estimated positions.
    
    Args:
        actual (dict): Actual position with x, y, z coordinates
        estimated (dict): Estimated position with x, y, z coordinates
        
    Returns:
        float: Euclidean distance
    """
    dx = actual['x'] - estimated['x']
    dy = actual['y'] - estimated['y']
    dz = actual['z'] - estimated['z']
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def adjust_reconstruction_data(data, x_offset=0.3, y_offset=0.3):
    """
    Adjust the reconstruction error data by subtracting offsets from coordinates.
    
    Args:
        data (dict): Reconstruction error data
        x_offset (float): Value to subtract from x coordinates
        y_offset (float): Value to subtract from y coordinates
        
    Returns:
        dict: Adjusted reconstruction error data
    """
    adjusted_data = data.copy()
    adjusted_comparisons = []
    
    # Process each position comparison
    for comp in data['position_comparisons']:
        adjusted_comp = comp.copy()
        
        # Adjust actual position
        adjusted_comp['actual_position'] = {
            'x': comp['actual_position']['x'] - x_offset,
            'y': comp['actual_position']['y'] - y_offset,
            'z': comp['actual_position']['z']
        }
        
        # Adjust estimated position
        adjusted_comp['estimated_position'] = {
            'x': comp['estimated_position']['x'] - x_offset,
            'y': comp['estimated_position']['y'] - y_offset,
            'z': comp['estimated_position']['z']
        }
        
        # Recalculate error based on new positions
        adjusted_comp['error'] = calculate_error(adjusted_comp['actual_position'], adjusted_comp['estimated_position'])
        
        adjusted_comparisons.append(adjusted_comp)
    
    # Update position comparisons
    adjusted_data['position_comparisons'] = adjusted_comparisons
    
    # Update total captures
    adjusted_data['total_captures'] = len(adjusted_comparisons)
    
    # Recalculate error statistics
    errors = [comp['error'] for comp in adjusted_comparisons]
    adjusted_data['error_statistics'] = {
        'min_error': min(errors),
        'max_error': max(errors),
        'mean_error': sum(errors) / len(errors),
        'median_error': sorted(errors)[len(errors) // 2],
        'std_error': np.std(errors)
    }
    
    # Update timestamp to indicate the data has been adjusted
    adjusted_data['analysis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return adjusted_data

def save_adjusted_data(adjusted_data, output_file):
    """
    Save adjusted reconstruction error data to a JSON file.
    
    Args:
        adjusted_data (dict): Adjusted reconstruction error data
        output_file (str): Path to output JSON file
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(adjusted_data, f, indent=4)
        print(f"Successfully saved adjusted reconstruction error data to {output_file}")
        
        # Print error statistics
        stats = adjusted_data['error_statistics']
        print("\nUpdated error statistics:")
        print(f"  Min error: {stats['min_error']:.6f} meters")
        print(f"  Max error: {stats['max_error']:.6f} meters")
        print(f"  Mean error: {stats['mean_error']:.6f} meters")
        print(f"  Median error: {stats['median_error']:.6f} meters")
        print(f"  Std error: {stats['std_error']:.6f} meters")
    except Exception as e:
        print(f"Error saving adjusted data: {e}")

def main():
    """Main function to process the reconstruction error file"""
    parser = argparse.ArgumentParser(description="Adjust reconstruction error data by subtracting offsets from coordinates")
    parser.add_argument("--input", default="/home/gabriel/Documents/tracker/4_camera_norm_tracker/reconstruction_error_analysis_floor.json",
                        help="Path to input reconstruction error JSON file")
    parser.add_argument("--output", default="/home/gabriel/Documents/tracker/4_camera_norm_tracker/reconstruction_error_analysis_floor_adjusted.json",
                        help="Path to output JSON file for adjusted data")
    parser.add_argument("--x-offset", type=float, default=0.3,
                        help="Value to subtract from x coordinates (default: 0.3)")
    parser.add_argument("--y-offset", type=float, default=0.3,
                        help="Value to subtract from y coordinates (default: 0.3)")
    
    args = parser.parse_args()
    
    # Load data from input file
    data = load_reconstruction_data(args.input)
    if not data:
        print("No data to process. Exiting.")
        return
    
    # Adjust coordinates
    print(f"Adjusting coordinates: x -= {args.x_offset}, y -= {args.y_offset}")
    adjusted_data = adjust_reconstruction_data(data, args.x_offset, args.y_offset)
    
    # Display a sample adjustment for verification
    if adjusted_data['position_comparisons']:
        original = data['position_comparisons'][0]
        adjusted = adjusted_data['position_comparisons'][0]
        
        print("\nSample adjustment (first entry):")
        print("Original actual position:")
        print(f"  X: {original['actual_position']['x']:.4f}")
        print(f"  Y: {original['actual_position']['y']:.4f}")
        print(f"  Z: {original['actual_position']['z']:.4f}")
        
        print("\nAdjusted actual position:")
        print(f"  X: {adjusted['actual_position']['x']:.4f}")
        print(f"  Y: {adjusted['actual_position']['y']:.4f}")
        print(f"  Z: {adjusted['actual_position']['z']:.4f}")
        
        print(f"\nOriginal error: {original['error']:.6f}")
        print(f"Adjusted error: {adjusted['error']:.6f}")
    
    # Save adjusted data
    save_adjusted_data(adjusted_data, args.output)
    print("Reconstruction error data adjustment complete.")

if __name__ == "__main__":
    main()
