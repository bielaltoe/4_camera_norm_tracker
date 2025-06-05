"""
Multi-View Person Tracking and 3D Reconstruction System

This script implements a real-time multi-camera tracking system that:
1. Processes video feeds from 4 cameras simultaneously
2. Detects and tracks people using YOLO
3. Matches detections across different views
4. Performs 3D triangulation
5. Visualizes results in real-time

Dependencies:
- OpenCV, NumPy, Matplotlib, NetworkX
- Custom modules: ploting_utils, video_loader, tracker, triangulation, matcher
"""
import argparse
import time
import logging
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import patheffects, cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

import json
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from ploting_utils import Utils
from video_loader import VideoLoader
from tracker import Tracker
from triangulation import triangulate_ransac
from config import YOLO_MODEL, CLASS_NAMES
from matcher import Matcher
from sort_3d_tracker import SORT_3D
import networkx as nx

# Import the newly created modules
from io_utils import save_3d_coordinates_with_ids
from visualization_utils import  draw_bbox, visualize_camera_positions
from graph_visualization import visualize_graph

# Set up matplotlib style for white background
plt.style.use('default')
PLOT_COLORS = list(plt.cm.tab10.colors) + list(plt.cm.Set1.colors)

def main():
    """
    Main execution function that orchestrates the multi-camera tracking pipeline.
    
    Pipeline steps:
    1. Load video streams
    2. Initialize tracking and matching components
    3. Process each frame:
       - Detect and track objects
       - Match detections across views
       - Perform 3D triangulation
       - Visualize results
    4. Save output video
    """

    argparser = argparse.ArgumentParser(description="Multi-Camera Tracking and 3D Reconstruction")
    argparser.add_argument("--video_path", type=str, default="videos", help="Path to video files")
    argparser.add_argument("--output_file", type=str, default="output.json", help="Output JSON file for 3D coordinates")
    argparser.add_argument("--save_coordinates", action="store_true", help="Save 3D coordinates to JSON file")
    argparser.add_argument("--use_sort", action="store_true", help="Use SORT algorithm for 3D tracking")
    argparser.add_argument("--max_age", type=int, default=10, help="Maximum frames object can be missing (SORT)")
    argparser.add_argument("--min_hits", type=int, default=3, help="Minimum hits to start tracking (SORT)")
    argparser.add_argument("--dist_threshold", type=float, default=1.0, help="Maximum distance for association (SORT)")
    argparser.add_argument("--class_list", type=int, nargs='+', default=[0], 
                          help="List of classes to track (e.g., 0 1 2) (default: 0 for person)")
    argparser.add_argument("--yolo_model", type=str, default=YOLO_MODEL, help="Path to YOLO model file")
    argparser.add_argument("--confidence", type=float, default=0.6, help="Confidence threshold for YOLO detection")
    argparser.add_argument("--distance_threshold", type=float, default=0.4, help="Distance threshold for matching") 
    argparser.add_argument("--drift_threshold", type=float, default=0.4, help="Drift threshold for matching")
    argparser.add_argument("--reference_point", type=str, default="bottom_center", 
                          choices=["bottom_center", "center", "top_center", "feet"],
                          help="Reference point on bounding box for triangulation")
    
    # Plot-related arguments (all enabled by default, disable with flags)
    argparser.add_argument("--headless", action="store_true", help="Disable all visualization (headless mode)")
    argparser.add_argument("--no-graph", action="store_true", help="Disable correspondence graph visualization")
    argparser.add_argument("--no-3d", action="store_true", help="Disable 3D plot visualization")
    argparser.add_argument("--no-video", action="store_true", help="Disable video mosaic visualization")
    argparser.add_argument("--save-video", action="store_true", help="Save output video regardless of visualization settings")
    argparser.add_argument("--output-video", type=str, default="output.mp4", help="Path to save the output video")
     
    # Arguments for exporting figures
    argparser.add_argument("--export_figures", action="store_true", help="Enable exporting of final plots (video mosaic, graph, 3D plot).")
    argparser.add_argument("--export_dpi", type=int, default=300, help="DPI for exported figures.")
    argparser.add_argument("--figures_output_dir", type=str, default="exported_figures", help="Directory to save exported figures.")

    args = argparser.parse_args()
    
    # Get arguments from parser
    video_path = args.video_path
    output_json_file = args.output_file
    save_flag = args.save_coordinates
    use_sort = args.use_sort
    max_age = args.max_age
    min_hits = args.min_hits
    dist_threshold = args.dist_threshold
    class_list = args.class_list
    yolo_model = args.yolo_model
    confidence = args.confidence
    distance_threshold = args.distance_threshold
    drift_threshold = args.drift_threshold
    reference_point = args.reference_point
    output_video_path = args.output_video
    
    # Handle visualization settings with opt-out approach
    show_plot = not args.headless
    show_graph = show_plot and not args.no_graph
    show_3d = show_plot and not args.no_3d
    show_video = show_plot and not args.no_video
    save_video = args.save_video
    
    # Arguments for exporting figures
    export_figures = args.export_figures
    export_dpi = args.export_dpi
    figures_output_dir = args.figures_output_dir

    logging.info(f"Video path: {video_path}")
    logging.info(f"Output file: {output_json_file}")
    logging.info(f"Using SORT 3D tracker: {use_sort}")
    logging.info(f"Visualization: Plot={show_plot}, Graph={show_graph}, 3D={show_3d}, Video={show_video}")
    logging.info(f"Reference point for triangulation: {reference_point}")
    
    if not os.path.exists(video_path):
        raise ValueError(f"Video path not found: {video_path}")

    video_files = os.listdir(video_path)
    logging.info(f"Video files: {video_files}")

    video_files = [os.path.join(video_path, f).replace("\\", "/") for f in video_files]
    logging.info(f"Video files: {video_files}")

    cam_numbers = [
        int(os.path.basename(cam).replace("cam", "").replace(".mp4", ""))
        for cam in video_files
    ]
    logging.info(f"Cam numbers: {cam_numbers}")

    # Create utils instance
    utils = Utils()
    video_loader = VideoLoader(video_files)
    tracker = Tracker([yolo_model for _ in range(len(cam_numbers))], cam_numbers, class_list, confidence)
    matcher = Matcher(distance_threshold, drift_threshold)
    
    # Initialize SORT 3D tracker if requested
    if use_sort:
        sort_tracker = SORT_3D(max_age=max_age, min_hits=min_hits, dist_threshold=dist_threshold)
        logging.info("SORT 3D tracker initialized")

    # Set up matplotlib figure with interactive backend and white background only if plotting is enabled
    video_ax = graph_ax = ax_3d = video_img = fig = None
    
    if show_plot:
        plt.ion()
        
        # Create figure with white background
        fig = plt.figure(figsize=(20, 12), facecolor='white')
        
        # Create a dynamic layout based on which plots are enabled
        if show_graph and show_3d and show_video:
            # Create a 2x2 layout with all plots
            gs = GridSpec(2, 2, figure=fig, width_ratios=[1.3, 1], wspace=0.05, hspace=0.1)
            
            # Video mosaic (left column spanning two rows)
            video_ax = fig.add_subplot(gs[:, 0])  # Spans both rows in first column
            
            # Graph visualization (top-right)
            graph_ax = fig.add_subplot(gs[0, 1])
            
            # 3D plot (bottom-right)
            ax_3d = fig.add_subplot(gs[1, 1], projection="3d", facecolor='white')
        elif show_3d and show_video:
            # Create a 1x2 layout without graph
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1], wspace=0.05)
            
            # Video mosaic (left column)
            video_ax = fig.add_subplot(gs[0, 0])
            
            # 3D plot (right column)
            ax_3d = fig.add_subplot(gs[0, 1], projection="3d", facecolor='white')
        elif show_graph and show_video:
            # Create a 1x2 layout without 3D
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1], wspace=0.05)
            
            # Video mosaic (left column)
            video_ax = fig.add_subplot(gs[0, 0])
            
            # Graph visualization (right column)
            graph_ax = fig.add_subplot(gs[0, 1])
        elif show_graph and show_3d:
            # Create a 1x2 layout with graph and 3D plots (no video)
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.05)
            
            # Graph visualization (left column)
            graph_ax = fig.add_subplot(gs[0, 0])
            
            # 3D plot (right column)
            ax_3d = fig.add_subplot(gs[0, 1], projection="3d", facecolor='white')
        elif show_video:
            # Only video
            video_ax = fig.add_subplot(111)
        elif show_3d:
            # Only 3D plot
            ax_3d = fig.add_subplot(111, projection="3d", facecolor='white')
        elif show_graph:
            # Only graph
            graph_ax = fig.add_subplot(111)
            
        # Configure axes if they exist
        if video_ax:
            video_ax.axis("off")
            video_img = video_ax.imshow(np.zeros((720, 1280, 3), dtype=np.uint8))
            video_title = video_ax.set_title("Multi-Camera View", 
                                            fontsize=16, color='black', fontweight='bold', pad=15)

        if ax_3d:
            # Configure 3D plot with white background styling - ONCE at initialization
            def configure_3d_axis(ax):
                """Configure 3D axis properties that need to be preserved"""
                ax.set_xlim([-4, 4])
                ax.set_ylim([-4, 4])
                ax.set_zlim([0, 4])
                
                # Set prominent axis labels that are clearly visible
                ax.set_xlabel("X (m)", fontsize=12, labelpad=13, color='black')
                ax.set_ylabel("Y (m)", fontsize=12, labelpad=13, color='black')
                ax.set_zlabel("Z (m)", fontsize=12, labelpad=13, color='black')
                
                # Set grid and tick colors for white background
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('lightgray')
                ax.yaxis.pane.set_edgecolor('lightgray')
                ax.zaxis.pane.set_edgecolor('lightgray')
                ax.tick_params(axis='x', colors='black', labelsize=10)
                ax.tick_params(axis='y', colors='black', labelsize=10)
                ax.tick_params(axis='z', colors='black', labelsize=10)
              
                ax.xaxis._axinfo["grid"]["color"] = 'lightgray'
                ax.yaxis._axinfo["grid"]["color"] = 'lightgray'
                ax.zaxis._axinfo["grid"]["color"] = 'lightgray'

            
            # Apply initial configuration
            configure_3d_axis(ax_3d)
            
            # Store the configuration function for reuse after clear()
            ax_3d._configure_func = configure_3d_axis

        # # Add a title to the figure if any plotting is enabled
        if show_plot:
            if reference_point == "center":
                ref_desc = "Center of bbox"
            elif reference_point == "top_center":
                ref_desc = "Top-center of bbox"
            elif reference_point == "feet":
                ref_desc = "20% above bottom-center (feet)"
            else:  # bottom_center
                ref_desc = "Bottom-center of bbox"
                
            fig.suptitle(f"Reference point: {ref_desc}", fontsize=15, color='black', fontweight='bold', y=0.98)

    pos = {}  # Position cache for graph nodes
    
    # Initialize video writer if saving video
    video_writer = None
    
    for frame_number in range(video_loader.get_number_of_frames()):
        graph = nx.Graph()
        frames = video_loader.get_frames()
        tracker.detect_and_track(frames)
        detections = tracker.get_detections()
        triangulated_points = []
        graph_component_ids = []
        node_color_map = {}
        
        # Process detections and build graph
        for d in detections:
            print(f"Detection ID: {d}")
            id = int(d.id)
            bbox = d.bbox
            cam = int(d.cam)
            frame = d.frame
            centroid = d.centroid
            name = d.name
            graph.add_node(
                f"cam{cam}id{id}",
                bbox=bbox,
                id=id,
                frame=frame,
                centroid=centroid,
                name=name,
            )

            # Get color based on ID for consistency
            color_rgb = utils.id_to_rgb_color(id)
            
            # Get class name string if available
            class_name = CLASS_NAMES.get(int(name), f"Class {int(name)}")
            
            # Draw stylish bounding box
            frame = draw_bbox(
                frame, 
                bbox, 
                class_name, 
                id, 
                color_rgb,
                reference_point  # Pass the reference point to the drawing function
            )

        # Match detections and build edges
        for k in cam_numbers:
            for j in cam_numbers:
                if k != j and k < j:
                    matches = matcher.match_detections(detections, [k, j])
                    for match in matches:
                        n1 = f"cam{k}id{int(match[0].id)}"
                        n2 = f"cam{j}id{int(match[1].id)}"
                        if n1 in graph.nodes and n2 in graph.nodes:
                            graph.add_edge(n1, n2)
        print("-\n" * 5)

        # Process triangulation
        class_ids = []  # Store class IDs for each triangulated point
        track_colors = {}  # Dictionary to store consistent colors by track ID
        
        for idx, c in enumerate(nx.connected_components(graph)):
            subgraph = graph.subgraph(c)
            if len(subgraph.nodes) > 1:
                ids = sorted(subgraph.nodes)
                d2_points = []
                proj_matricies = []
                # Extract class from first node (all nodes in component should have same class)
                class_id = subgraph.nodes[ids[0]]["name"]

                for node in ids:
                    cam = int(node.split("cam")[1].split("id")[0])
                    id = int(node.split("id")[1])
                    centroid = subgraph.nodes[node]["centroid"]
                    bbox = subgraph.nodes[node]["bbox"]
                    P_cam = matcher.P_all[cam]
                    
                    # Get reference point based on user selection
                    if reference_point == "center":
                        # Use the center of the bounding box
                        point_2d = ((bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2)
                    elif reference_point == "top_center":
                        # Use the top-center point
                        point_2d = ((bbox[2]+bbox[0])/2, bbox[1])
                    elif reference_point == "feet":
                        # Use the bottom-center point but offset slightly from the edge
                        # This helps with cases where the bottom of the bounding box is slightly below the feet
                        bottom_offset = 0.2 * (bbox[3] - bbox[1])  # 20% offset from bottom
                        point_2d = ((bbox[2]+bbox[0])/2, bbox[3] - bottom_offset)
                    else:  # Default to bottom_center
                        # Use the bottom-center point
                        point_2d = ((bbox[2]+bbox[0])/2, bbox[3])
                    
                    d2_points.append(point_2d)
                    proj_matricies.append(P_cam)

                if len(d2_points) >= 2:
                    point_3d, _ = triangulate_ransac(
                        proj_matricies, d2_points
                    )
                    print(f"3D point: {point_3d}")
                    triangulated_points.append(point_3d)
                    graph_component_ids.append(idx)
                    class_ids.append(class_id)  # Store class ID with triangulated point
                    
                    # Use a temporary ID for color mapping - this will be corrected after SORT
                    temp_color_rgb = utils.id_to_rgb_color(idx)
                    track_colors[idx] = temp_color_rgb
                    
                    for node in subgraph.nodes:
                        node_color_map[node] = utils.normalize_rgb_color(temp_color_rgb)
        
        # If we're using SORT, update the 3D tracker
        if use_sort and triangulated_points:
            # Update the SORT tracker with triangulated points AND their classes
            sort_result = sort_tracker.update(triangulated_points, class_ids)
            
            # Use the SORT-tracked versions directly
            point_3d_list = sort_result['positions']
            track_ids = sort_result['ids']
            sorted_class_ids = sort_result['class_ids']  # Classes are now maintained by SORT
            trajectories = sort_result['trajectories']
            
            # CRITICAL FIX: Update colors based on actual track IDs from SORT
            # Clear previous temporary colors
            track_colors.clear()
            node_color_map.clear()
            
            # Assign colors based on SORT track IDs for consistency
            for i, (point_3d, track_id) in enumerate(zip(point_3d_list, track_ids)):
                color_rgb = utils.id_to_rgb_color(track_id)
                track_colors[track_id] = color_rgb
                
                # Update node colors for graph visualization
                # Find the corresponding graph nodes for this point
                if i < len(triangulated_points):
                    # Match the SORT point back to original triangulated points to find graph nodes
                    for comp_idx, orig_point in enumerate(triangulated_points):
                        # Check if this SORT point corresponds to this original point
                        distance = np.linalg.norm(np.array(point_3d) - np.array(orig_point))
                        if distance < 0.1:  # Small threshold for matching
                            # Find nodes in the corresponding graph component
                            components = list(nx.connected_components(graph))
                            if comp_idx < len(components):
                                component_nodes = components[comp_idx]
                                for node in component_nodes:
                                    node_color_map[node] = utils.normalize_rgb_color(color_rgb)
                            break
            
            logging.info(f"Frame {frame_number}: SORT tracking {len(point_3d_list)} objects")
            
        else:
            # Without SORT, use raw triangulated points
            point_3d_list = triangulated_points
            track_ids = graph_component_ids
            trajectories = {}
            sorted_class_ids = class_ids
            
            # For non-SORT mode, colors are already assigned correctly above

        # Save coordinates if requested
        if save_flag and point_3d_list:
            if use_sort:
                # Save with track IDs and class IDs for consistent tracking
                save_3d_coordinates_with_ids(frame_number, point_3d_list, track_ids, output_json_file, sorted_class_ids)
            else:
                # For backward compatibility, use the original function when not using SORT
                save_3d_coordinates_with_ids(frame_number, point_3d_list, track_ids, output_json_file, class_ids)
        elif save_flag:
            logging.info(f"No 3D points detected for frame {frame_number}")

        # Create video mosaic with annotations and better formatting if video visualization is enabled
        if show_plot and show_video:
            processed_frames = []
            for idx, frame in enumerate(frames):
                # Convert BGR to RGB for matplotlib
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Add a nice camera label
                h, w = rgb_frame.shape[:2]
                overlay = rgb_frame.copy()
                cv2.rectangle(overlay, (0, 0), (175, 40), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, rgb_frame, 0.3, 0, rgb_frame)
                cv2.putText(rgb_frame, f"Camera {idx}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                processed_frames.append(cv2.resize(rgb_frame, (540, 360)))

            # Create video mosaic with a small border between frames
            border = np.ones((360, 5, 3), dtype=np.uint8) * 255  # White vertical border
            h_border = np.ones((5, 1085, 3), dtype=np.uint8) * 255  # White horizontal border
            
            top_row = np.hstack((processed_frames[0], border, processed_frames[1]))
            bottom_row = np.hstack((processed_frames[2], border, processed_frames[3]))
            full_mosaic = np.vstack((top_row, h_border, bottom_row))
            
            # Add frame counter to the video mosaic
            cv2.rectangle(full_mosaic, (full_mosaic.shape[1]-200, full_mosaic.shape[0]-50), 
                        (full_mosaic.shape[1], full_mosaic.shape[0]), (0, 0, 0), -1)
            cv2.putText(full_mosaic, f"Frame: {frame_number}", (full_mosaic.shape[1]-190, full_mosaic.shape[0]-11), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            video_img.set_data(full_mosaic)
            video_title.set_text(f"Multi-Camera View - {len(point_3d_list)} Objects Detected")

        # Update 3D plot with better styling
        if show_plot and show_3d:
            ax_3d.clear()
            
            # Efficiently restore axis configuration using stored function
            if hasattr(ax_3d, '_configure_func'):
                ax_3d._configure_func(ax_3d)
            else:
                # Fallback: simple axis labels only
                ax_3d.set_xlabel("X (m)", fontsize=11, labelpad=12, color='black')
                ax_3d.set_ylabel("Y (m)", fontsize=11, labelpad=12, color='black')
                ax_3d.set_zlabel("Z (m)", fontsize=11, labelpad=12, color='black')
                ax_3d.tick_params(axis='x', colors='black', labelsize=10)
                ax_3d.tick_params(axis='y', colors='black', labelsize=10)
                ax_3d.tick_params(axis='z', colors='black', labelsize=10)
            
            # Add ground plane and enhanced camera positions for better spatial understanding
            if hasattr(matcher, 'P_all'):
                try:
                    visualize_camera_positions(ax_3d, matcher.P_all)
                except Exception as e:
                    logging.warning(f"Error visualizing camera positions: {str(e)}")
            
            # Collect legend information
            legend_elements = []
            
            # Plot each object with improved styling
            for point_idx, point_3d in enumerate(point_3d_list):
                track_id = track_ids[point_idx] if point_idx < len(track_ids) else point_idx
                
                # Get class info if available
                class_id = None
                if sorted_class_ids and point_idx < len(sorted_class_ids):
                    class_id = sorted_class_ids[point_idx]
                
                # Use consistent color based on track ID
                if track_id in track_colors:
                    color_rgb = track_colors[track_id]
                    color_rgb_norm = utils.normalize_rgb_color(color_rgb)
                else:
                    # Fallback color using existing approach
                    color_idx = track_id % len(PLOT_COLORS)
                    color_rgb_norm = PLOT_COLORS[color_idx]
                
                # Create legend label with ID, class, and height information
                if class_id is not None:
                    try:
                        class_name = CLASS_NAMES.get(int(class_id), "Unknown")
                        legend_label = f"ID {track_id}: {class_name} (h: {point_3d[2]:.2f}m)"
                    except:
                        legend_label = f"ID {track_id} (h: {point_3d[2]:.2f}m)"
                else:
                    legend_label = f"ID {track_id} (h: {point_3d[2]:.2f}m)"
                
                # Add object marker with class information (no individual text labels)
                scatter = ax_3d.scatter(
                    point_3d[0],
                    point_3d[1],
                    point_3d[2],
                    c=[color_rgb_norm],
                    s=150,
                    marker='o',
                    edgecolors='black',
                    linewidths=1.5,
                    alpha=0.8,
                    zorder=10,
                    label=legend_label  # Add to legend instead of text on plot
                )
                
                # Add to legend elements for custom legend positioning
                from matplotlib.lines import Line2D
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color_rgb_norm, markersize=10,
                                            markeredgecolor='black', markeredgewidth=1.5,
                                            label=legend_label))
                
                # Plot trajectory with consistent color
                if use_sort and track_id in trajectories and len(trajectories[track_id]) > 1:
                    trajectory = np.array(trajectories[track_id])
                    ax_3d.plot(
                        trajectory[:, 0],
                        trajectory[:, 1],
                        trajectory[:, 2],
                        c=color_rgb_norm,
                        alpha=0.8,
                        linewidth=2.5,
                        zorder=5
                    )
                    
                    # Add vertical line with consistent color
                    ax_3d.plot(
                        [point_3d[0], point_3d[0]],
                        [point_3d[1], point_3d[1]], 
                        [0, point_3d[2]],
                        '--', 
                        color=color_rgb_norm,
                        alpha=0.5, 
                        linewidth=1,
                        zorder=4
                    )
            
            # Configure 3D plot appearance
            ax_3d.set_xlim([-4, 4])
            ax_3d.set_ylim([-4, 4])
            ax_3d.set_zlim([0, 4])
            
            # Add title with tracking information
            title_text = f"3D Position - Frame {frame_number}"
            if use_sort:
                title_text += f" - SORT Tracking ({len(point_3d_list)} objects)"
            else:
                title_text += f" - Raw Triangulation ({len(point_3d_list)} objects)"
                
            ax_3d.set_title(
                title_text,
                fontsize=14,
                color='black',
                fontweight='bold',
                pad=3
            )
            
            # Add custom legend positioned outside the plot area for clarity
            if legend_elements:
                # Position legend to the upper left corner of the plot
                legend = ax_3d.legend(handles=legend_elements, 
                                    loc='upper left', 
                                    bbox_to_anchor=(0.01, 0.99),  # Adjust anchor to be slightly inside the plot
                                    fontsize=10, # Adjusted fontsize slightly for better fit
                                    title="Tracked Objects",
                                    title_fontsize=12) # Adjusted title fontsize
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)
                legend.get_frame().set_edgecolor('lightgray')

        # Update graph visualization if requested
        if show_plot and show_graph and graph_ax is not None:
            # Update node positions for persistent layout
            for node in graph.nodes():
                if node not in pos:
                    cam = int(node.split("cam")[1].split("id")[0])
                    obj_id = int(node.split("id")[1])
                    x = (cam - 1) * 6  # Increased horizontal spacing
                    y = obj_id * 3     # Increased vertical spacing
                    pos[node] = (x, y)
                    
            # Visualize the graph
            visualize_graph(graph, graph_ax, frame_number, pos, node_color_map)

        # Update figure and handle events if plotting is enabled
        if show_plot:
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Small delay for smooth visualization
            plt.pause(0.1)
            final_frame = utils.fig_to_image(fig)
            
            # Save the video frame if requested
            if save_video or (show_plot and frame_number == 0):
                # Initialize the video writer if it hasn't been done yet
                if video_writer is None:
                    video_writer = cv2.VideoWriter(
                        output_video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (final_frame.shape[1], final_frame.shape[0]),
                    )
                
                final_frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                video_writer.write(final_frame_rgb)
            
            # Check for exit condition if plotting is enabled
            if not plt.fignum_exists(fig.number):
                break
        elif frame_number % 10 == 0:  # Report progress periodically when not plotting
            logging.info(f"Processing frame {frame_number}/{video_loader.get_number_of_frames()}")

    # Cleanup
    if show_plot:
        plt.ioff()
        plt.close()
    
    if video_writer is not None:
        video_writer.release()
    
    video_loader.release()
    logging.info("Processing complete")

    # Export figures if requested
    if export_figures and fig is not None:
        if not os.path.exists(figures_output_dir):
            os.makedirs(figures_output_dir)
        
        # Get the base name of the video path to use in filenames
        video_basename = os.path.basename(os.path.normpath(video_path))

        # Save the entire figure (mosaic, graph, 3D plot)
        # Ensure plots are up-to-date before saving
        fig.canvas.draw()

        # Save video mosaic part if available
        if show_video and video_ax is not None:
            # Create a temporary figure for the video mosaic
            temp_fig_video, temp_ax_video = plt.subplots(figsize=(full_mosaic.shape[1]/100, full_mosaic.shape[0]/100), facecolor='white')
            temp_ax_video.imshow(full_mosaic)
            temp_ax_video.axis('off')
            mosaic_filename = os.path.join(figures_output_dir, f"{video_basename}_video_mosaic_last_frame.png")
            temp_fig_video.savefig(mosaic_filename, dpi=export_dpi, bbox_inches='tight', pad_inches=0)
            plt.close(temp_fig_video)
            logging.info(f"Video mosaic saved to {mosaic_filename} at {export_dpi} DPI")

        # Save graph part if available
        if show_graph and graph_ax is not None:
            # Create a temporary figure for the graph
            # Extent of the graph_ax
            bbox = graph_ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
            temp_fig_graph = plt.figure(figsize=(bbox.width, bbox.height), facecolor='white')
            temp_ax_graph = temp_fig_graph.add_subplot(111)
            
            # Redraw the graph content onto the temporary axis
            # This requires access to the graph drawing function and its parameters
            # For simplicity, we\'ll save the relevant part of the existing figure if possible
            # or notify the user that this specific part needs a more direct save method.
            # The ideal way is to have a function that just draws the graph on a given ax.
            # For now, we save the graph_ax portion from the main figure.
            graph_filename = os.path.join(figures_output_dir, f"{video_basename}_graph_last_frame.png")
            fig.savefig(graph_filename, dpi=export_dpi, bbox_inches=bbox, pad_inches=0)
            plt.close(temp_fig_graph) # Close the temporary figure
            logging.info(f"Graph saved to {graph_filename} at {export_dpi} DPI")

        # Save 3D plot part if available
        if show_3d and ax_3d is not None:
            # Create a temporary figure for the 3D plot
            # Extent of the ax_3d
            bbox = ax_3d.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
            temp_fig_3d = plt.figure(figsize=(bbox.width, bbox.height), facecolor='white') # Ensure background is white
            # Copy the 3D plot to the new figure - this is tricky with 3D plots.
            # A common way is to re-plot. For simplicity, we save the ax_3d portion.
            plot3d_filename = os.path.join(figures_output_dir, f"{video_basename}_3d_plot_last_frame.png")
            fig.savefig(plot3d_filename, dpi=export_dpi, bbox_inches=bbox, pad_inches=0)
            plt.close(temp_fig_3d) # Close the temporary figure
            logging.info(f"3D plot saved to {plot3d_filename} at {export_dpi} DPI")

if __name__ == "__main__":
    main()