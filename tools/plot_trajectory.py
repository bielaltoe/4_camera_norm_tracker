import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import argparse
from collections import defaultdict

def load_trajectory_data(json_file):
    """Load trajectory data from a JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def extract_trajectories(data):
    """Extract trajectories from the loaded data, now with class information."""
    frames = len(data)
    
    # Track unique IDs and classes to support class-based visualization
    unique_ids = set()
    for frame_data in data:
        for point in frame_data.get('points', []):
            if isinstance(point, dict) and 'id' in point:
                unique_ids.add(point['id'])
    
    # Sort IDs for consistent indexing
    id_list = sorted(list(unique_ids))
    id_to_idx = {id: idx for idx, id in enumerate(id_list)}
    max_points = len(id_list) if id_list else 0
    
    # Initialize trajectories array with NaNs for potential missing points
    trajectories = np.full((frames, max_points, 3), np.nan)
    classes = np.full(max_points, -1)  # Default class as -1 (unknown)
    timestamps = []
    frame_numbers = []
    
    for i, frame_data in enumerate(data):
        timestamps.append(frame_data.get('timestamp', f"Frame {i}"))
        frame_numbers.append(frame_data.get('frame', i))
        
        for point in frame_data.get('points', []):
            if isinstance(point, dict) and 'id' in point and 'position' in point:
                obj_id = point['id']
                if obj_id in id_to_idx:
                    idx = id_to_idx[obj_id]
                    
                    # Extract position data
                    if isinstance(point['position'], list):
                        trajectories[i, idx] = point['position']
                    
                    # Extract class data if available
                    if 'class' in point and classes[idx] == -1:
                        classes[idx] = point['class']
    
    return trajectories, timestamps, frame_numbers, classes, id_list

def get_class_name(class_id):
    """Convert class ID to name based on COCO classes."""
    class_names = {
        0: 'person',
        56: 'chair',
        # Add more class mappings as needed
    }
    return class_names.get(class_id, f"class_{class_id}")

def plot_trajectories(trajectories, timestamps, frame_numbers, classes, id_list, output_file=None):
    """Create a 3D plot of the trajectories with class-based styling."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Class-based color scheme
    class_colors = {
        0: 'tab:blue',      # Person - blue
        56: 'tab:red',      # Chair - red
        -1: 'tab:gray'      # Unknown - gray
    }
    
    # Class-based markers
    class_markers = {
        0: 'o',      # Person - circle
        56: 's',     # Chair - square
        -1: 'x'      # Unknown - x
    }
    
    # Find min and max coordinate values for axis limits
    valid_points = trajectories[~np.isnan(trajectories)]
    if len(valid_points) > 0:
        min_val = np.min(valid_points)
        max_val = np.max(valid_points)
        buffer = (max_val - min_val) * 0.1  # Add a 10% buffer
        ax_min = min_val - buffer
        ax_max = max_val + buffer
    else:
        ax_min, ax_max = -2, 2
    
    # Initialize trajectory lines and point markers
    lines = []
    points = []
    num_trajectories = trajectories.shape[1]
    
    # Group by class for the legend
    class_handles = {}
    
    for i in range(num_trajectories):
        obj_id = id_list[i]
        class_id = int(classes[i])
        color = class_colors.get(class_id, f"C{i % 10}")
        marker = class_markers.get(class_id, 'o')
        
        label = f"{get_class_name(class_id)} ID:{obj_id}"
        
        line, = ax.plot([], [], [], '.-', color=color, alpha=0.7, linewidth=2)
        point, = ax.plot([], [], [], marker=marker, markersize=10, color=color, label=label)
        
        lines.append(line)
        points.append(point)
        
        # Store one handle per class for the legend
        if class_id not in class_handles:
            class_handles[class_id] = point
    
    # Plot title with timestamp information
    title = ax.set_title(f'Frame: 0, Time: {timestamps[0] if timestamps else "N/A"}')
    
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_zlim(ax_min, ax_max)
    
    # Add a grid for better depth perception
    ax.grid(True)
    
    # Add a legend by unique classes
    handles = list(class_handles.values())
    labels = [h.get_label() for h in handles]
    ax.legend(handles=handles, labels=labels, loc='upper right')
    
    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        return lines + points + [title]
    
    def animate(frame):
        # Update title first
        if frame < len(timestamps):
            title.set_text(f'Frame: {frame_numbers[frame]}, Time: {timestamps[frame]}')
        
        # Update all trajectories
        for i, (line, point) in enumerate(zip(lines, points)):
            # Get all points up to current frame for trajectory
            x = trajectories[:frame+1, i, 0]
            y = trajectories[:frame+1, i, 1]
            z = trajectories[:frame+1, i, 2]
            
            # Remove NaN values for plotting
            valid = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
            x_valid = x[valid]
            y_valid = y[valid]
            z_valid = z[valid]
            
            # Update trajectory line
            line.set_data(x_valid, y_valid)
            line.set_3d_properties(z_valid)
            
            # Update current position point
            if frame < len(trajectories) and not np.isnan(trajectories[frame, i]).any():
                point.set_data([trajectories[frame, i, 0]], [trajectories[frame, i, 1]])
                point.set_3d_properties([trajectories[frame, i, 2]])
            else:
                point.set_data([], [])
                point.set_3d_properties([])
                
        return lines + points + [title]
    
    frames = len(trajectories)
    ani = FuncAnimation(fig, animate, frames=frames, init_func=init, 
                        interval=100, blit=True)
    
    # Improve layout
    plt.tight_layout()
    
    if output_file:
        print(f"Saving animation to {output_file}...")
        ani.save(output_file, writer='ffmpeg')
        print("Animation saved successfully!")
        # Also show the plot after saving if desired
        plt.show()
    else:
        plt.show()

def create_static_plot(trajectories, classes, id_list, output_file=None):
    """Create a static 3D plot showing complete trajectories."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Class-based color scheme
    class_colors = {
        0: 'tab:blue',      # Person - blue
        56: 'tab:red',      # Chair - red
        -1: 'tab:gray'      # Unknown - gray
    }
    
    # Class-based markers
    class_markers = {
        0: 'o',      # Person - circle
        56: 's',     # Chair - square
        -1: 'x'      # Unknown - x
    }
    
    # Process each trajectory
    for i in range(trajectories.shape[1]):
        obj_id = id_list[i]
        class_id = int(classes[i])
        color = class_colors.get(class_id, f"C{i % 10}")
        marker = class_markers.get(class_id, 'o')
        
        # Get trajectory data
        x = trajectories[:, i, 0]
        y = trajectories[:, i, 1]
        z = trajectories[:, i, 2]
        
        # Remove NaN values
        valid = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
        x_valid = x[valid]
        y_valid = y[valid]
        z_valid = z[valid]
        
        if len(x_valid) > 0:
            # Plot trajectory line
            ax.plot(x_valid, y_valid, z_valid, '-', color=color, alpha=0.7, linewidth=2)
            
            # Plot start and end points with different markers
            ax.scatter(x_valid[0], y_valid[0], z_valid[0], s=100, color=color, marker='o', label=f"Start {get_class_name(class_id)} ID:{obj_id}")
            ax.scatter(x_valid[-1], y_valid[-1], z_valid[-1], s=100, color=color, marker='s', label=f"End {get_class_name(class_id)} ID:{obj_id}")
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add a grid for better depth perception
    ax.grid(True)
    
    # Add title
    ax.set_title('3D Trajectories')
    
    # Handle duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    
    if output_file:
        output_static = os.path.splitext(output_file)[0] + '_static.png'
        plt.savefig(output_static, dpi=300, bbox_inches='tight')
        print(f"Static plot saved to {output_static}")
    
    return fig, ax

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot 3D trajectories from JSON data')
    parser.add_argument('--file', '-f', type=str, help='Path to the JSON file containing trajectory data')
    parser.add_argument('--output', '-o', type=str, help='Path to save the animation (optional)')
    parser.add_argument('--static', '-s', action='store_true', help='Generate a static plot of complete trajectories')
    parser.add_argument('--save-only', action='store_true', help='Only save the animation without displaying it')
    parser.add_argument('--class-filter', '-c', type=int, nargs='+', help='Filter by class IDs (e.g., 0 56)')
    args = parser.parse_args()
    
    # Use the provided file path or search for teste0.json
    if args.file:
        json_file = args.file
        if not os.path.isfile(json_file):
            raise FileNotFoundError(f"Could not find file: {json_file}")
    else:
        # Search for known JSON outputs in likely locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_files = [
            os.path.join(script_dir, 'teste0.json'),
            os.path.join(script_dir, 'output.json'),
            os.path.join(script_dir, 'rastro_pessoa.json'),
            os.path.join(script_dir, 'experiments/three_chairs/teste_tracking_kalman.json')
        ]
        
        for potential_file in potential_files:
            if os.path.isfile(potential_file):
                json_file = potential_file
                break
        else:
            print("Could not find trajectory data file automatically.")
            print("Please specify the file path using --file argument.")
            return
    
    print(f"Loading data from {json_file}")
    data = load_trajectory_data(json_file)
    trajectories, timestamps, frame_numbers, classes, id_list = extract_trajectories(data)
    
    # Apply class filtering if requested
    if args.class_filter:
        class_mask = np.isin(classes, args.class_filter)
        trajectories = trajectories[:, class_mask, :]
        classes = classes[class_mask]
        id_list = [id for i, id in enumerate(id_list) if class_mask[i]]
        
        if len(id_list) == 0:
            print(f"No trajectories found for classes: {args.class_filter}")
            return
        
        print(f"Filtered to show only classes: {args.class_filter}")
    
    # Generate static plot if requested
    if args.static:
        static_output = args.output.replace('.mp4', '_static.png') if args.output else None
        create_static_plot(trajectories, classes, id_list, static_output)
    
    # If save-only flag is set and output file is specified, only save without showing
    if args.save_only and args.output:
        print(f"Saving animation to {args.output} without displaying...")
        plot_trajectories(trajectories, timestamps, frame_numbers, classes, id_list, args.output)
        print("Animation saved successfully!")
    else:
        plot_trajectories(trajectories, timestamps, frame_numbers, classes, id_list, args.output)

if __name__ == "__main__":
    main()
