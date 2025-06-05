import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import json

def generate_grid(center_x=0.3, center_y=0.3, grid_size=7, spacing=0.5):
    """
    Generate a grid centered at a specific point with given size and spacing.
    
    Parameters:
    - center_x, center_y: Coordinates of the center of the grid
    - grid_size: Number of points in each dimension
    - spacing: Distance between adjacent grid points in meters
    
    Returns:
    - X, Y: Arrays containing the x and y coordinates
    """
    # Calculate grid ranges
    offset = (grid_size - 1) / 2 * spacing
    x_range = np.linspace(center_x - offset, center_x + offset, grid_size)
    y_range = np.linspace(center_y - offset, center_y + offset, grid_size)
    
    # Create meshgrid
    X, Y = np.meshgrid(x_range, y_range)
    
    return X, Y

# Copied from get_pictures.py
def camera_parameters(file):
    camera_data = json.load(open(file))
    K = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)
    res = [camera_data['resolution']['width'],
           camera_data['resolution']['height']]
    # Fix: Access extrinsic as an array
    tf = np.array(camera_data['extrinsic'][0]['tf']['doubles']).reshape(4, 4)
    
    # Get the RT matrix (original camera-to-world transformation)
    R = tf[:3, :3]
    T = tf[:3, 3].reshape(3, 1)
    
    # Compute the inverse transformation (world-to-camera)
    # For a rotation matrix, its inverse is its transpose
    R_inv = R.transpose()
    # For translation, we need to apply -T rotated by inverse R
    T_inv = -R_inv @ T
    
    dis = np.array(camera_data['distortion']['doubles'])
    return K, R, T, res, dis, R_inv, T_inv

# Copied from get_pictures.py
def plot_camera_axes(R, T, ax, scale=0.5, label=''):
    """
    Plota os eixos do referencial da câmera.
    - R: matriz de rotação (3x3), onde cada coluna representa um eixo (x, y, z)
    - T: vetor de translação (3x1), posição da câmera no ambiente
    - ax: objeto Axes3D onde será plotado
    - scale: comprimento das setas para visualização
    - label: rótulo para identificar a câmera
    """
    # Converte T para array 1D
    origin = T.flatten()
    
    # Vetores dos eixos da câmera em coordenadas globais
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    
    # Plota os eixos usando ax.quiver
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              color='r', length=scale, normalize=True)
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              color='g', length=scale, normalize=True)
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='b', length=scale, normalize=True)
    
    # Adiciona um rótulo na posição da câmera
    ax.text(origin[0], origin[1], origin[2], label, fontsize=10)

def plot_grid(X, Y, spacing=0.5, cameras_data=None, camera_indices_to_plot=None):
    """
    Plot the grid in 3D with points at Z=0 and camera axes.
    
    Parameters:
    - X, Y: Arrays containing the x and y coordinates of the grid
    - spacing: Distance between points for title display
    - cameras_data: Dictionary containing calibration data for cameras
    - camera_indices_to_plot: List of camera indices to plot
    """
    grid_size = X.shape[0]
    center_x = X[grid_size//2, grid_size//2]
    center_y = Y[grid_size//2, grid_size//2]

    # Create 3D plot with larger figure size
    fig = plt.figure(figsize=(27, 27))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot grid points at Z=0
    ax.scatter(X, Y, np.zeros_like(X), color='blue', s=50, label='Grid Points (Z=0)')
    
    # Plot world origin
    ax.scatter(0, 0, 0, color='k', s=100, marker='x', label='World Origin')
    
    # Plot camera axes - exactly like in get_pictures.py
    if cameras_data and camera_indices_to_plot:
        for cam_idx in camera_indices_to_plot:
            if cam_idx in cameras_data:
                plot_camera_axes(
                    cameras_data[cam_idx]['R_inv'], 
                    cameras_data[cam_idx]['T_inv'], 
                    ax, scale=0.5, 
                    label=f'Câmera {cam_idx}'
                )
            else:
                print(f"Warning: Calibration data for camera {cam_idx} not found.")

    # Set axis labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'{grid_size}x{grid_size} Grid (Z=0) & Camera Poses\nSpacing: {spacing}m, Center: ({center_x:.1f}, {center_y:.1f})', fontsize=14)
    
    # Set reasonable axis limits
    all_x = list(X.flatten())
    all_y = list(Y.flatten())
    all_z = [0]  # Grid is at Z=0
    
    if cameras_data and camera_indices_to_plot:
        for cam_idx in camera_indices_to_plot:
            if cam_idx in cameras_data:
                T_cam = cameras_data[cam_idx]['T_inv'].flatten()
                all_x.append(T_cam[0])
                all_y.append(T_cam[1])
                all_z.append(T_cam[2])
    
    if all_x and all_y and all_z:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(0, 3)

    # Set axis ticks with 0.5m spacing
    x_ticks = np.arange(-3, 3.5, 0.5)
    y_ticks = np.arange(-3, 3.5, 0.5)
    z_ticks = np.arange(0, 3.5, 0.5)
    
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    
    # Adjust tick label font size
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)

    ax.legend()
    plt.tight_layout()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate and plot a 3D grid of points with camera poses.')
    parser.add_argument('--center-x', type=float, default=0.3,
                        help='X-coordinate of the center (default: 0.3)')
    parser.add_argument('--center-y', type=float, default=0.3,
                        help='Y-coordinate of the center (default: 0.3)')
    parser.add_argument('--size', type=int, default=7,
                        help='Grid size (number of points in each dimension, default: 7)')
    parser.add_argument('--spacing', type=float, default=0.5,
                        help='Spacing between points in meters (default: 0.5)')
    parser.add_argument('--no-coords', action='store_true',
                        help='Hide coordinate labels (Note: 3D plot does not show individual point coords by default)')
    parser.add_argument('--cameras', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='List of camera indices to plot (e.g., 0 1 2 3)')
    parser.add_argument('--calib-path', type=str, default='calibrations',
                        help='Path to the directory containing camera calibration JSON files')
    args = parser.parse_args()
    
    # Generate the grid coordinates
    X, Y = generate_grid(args.center_x, args.center_y, args.size, args.spacing)

    # Load camera parameters
    cameras_data = {}
    camera_indices_to_plot = args.cameras
    if camera_indices_to_plot:
        print(f"Loading calibration data for cameras: {camera_indices_to_plot}")
        for cam_idx in camera_indices_to_plot:
            json_file = f"{args.calib_path}/{cam_idx}.json"
            try:
                K, R, T, res, dis, R_inv, T_inv = camera_parameters(json_file)
                cameras_data[cam_idx] = {
                    'K': K, 'R': R, 'T': T, 'res': res, 'dis': dis, 
                    'R_inv': R_inv, 'T_inv': T_inv
                }
                print(f"Loaded calibration for camera {cam_idx}")
            except FileNotFoundError:
                print(f"Error: Calibration file {json_file} not found for camera {cam_idx}.")
            except Exception as e:
                print(f"Error loading calibration for camera {cam_idx} from {json_file}: {e}")
    
    # Plot the grid and cameras
    plot_grid(X, Y, args.spacing, cameras_data, camera_indices_to_plot)
    
    # Print grid information
    print(f"Generated a {args.size}x{args.size} grid centered at ({args.center_x}, {args.center_y}) with {args.spacing}m spacing")
    
    # Print the coordinates in a tabular format
    print("\nGrid Coordinates:")
    for i in range(args.size):
        for j in range(args.size):
            print(f"Point ({i},{j}): ({X[i, j]:.2f}, {Y[i, j]:.2f})")
    
    # Show the plot or save for publication
    plt.show()
    # For an IEEE paper, save the figure with high DPI:
    plt.savefig("grid_plot_3d.png", dpi=300) 
    # plt.savefig("grid_plot_3d.pdf", format='pdf') # Vector format is often preferred

if __name__ == "__main__":
    main()
