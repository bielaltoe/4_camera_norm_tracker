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

import logging
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from ploting_utils import Utils
from video_loader import VideoLoader
from tracker import Tracker
from triangulation import triangulate_point_from_multiple_views_linear
from config import YOLO_MODEL
from matcher import Matcher
import networkx as nx


def visualize_graph(graph, ax, frame_number, pos, node_color_map):
    """
    Visualizes the detection graph showing connections between cameras and tracked objects.
    
    Args:
        graph (nx.Graph): NetworkX graph containing detection information
        ax (matplotlib.axes.Axes): Matplotlib axis for drawing
        frame_number (int): Current frame number for title
        pos (dict): Node positions for visualization
        node_color_map (dict): Mapping of nodes to their display colors
    """
    ax.clear()
    
    # Draw nodes with colors from the 3D point mapping
    node_colors = [node_color_map.get(node, (0.5, 0.5, 0.5)) for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_size=700, ax=ax, node_color=node_colors)
    
    # Draw edges with weights
    edge_labels = nx.get_edge_attributes(graph, "distance")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, ax=ax)
    
    # Auto-adjust axis limits
    x_vals = [pos[node][0] for node in graph.nodes()]
    y_vals = [pos[node][1] for node in graph.nodes()]
    ax.set_xlim(min(x_vals)-2, max(x_vals)+2)
    ax.set_ylim(min(y_vals)-2, max(y_vals)+2)
    ax.set_title(f"Frame {frame_number}")
    
    
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
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        logging.info(f"Video path: {video_path}")
    else:
        video_path = "videos"
        if not os.path.exists(video_path):
            raise ValueError("No video path provided")

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
    tracker = Tracker([YOLO_MODEL for _ in range(len(cam_numbers))], cam_numbers)
    matcher = Matcher()

    # Set up matplotlib figure with interactive backend
    plt.ion()
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 2, figure=fig)  # 2 rows, 2 columns

    # Video mosaic (left column)
    video_ax = fig.add_subplot(gs[:, 0])
    video_ax.axis("off")
    video_img = video_ax.imshow(np.zeros((720, 1280, 3), dtype=np.uint8))

    # Visualization plots (right column)
    graph_ax = fig.add_subplot(gs[0, 1])
    ax_3d = fig.add_subplot(gs[1, 1], projection="3d")

    # Configure 3D plot
    ax_3d.set_xlim([-4, 4])
    ax_3d.set_ylim([-4, 4])
    ax_3d.set_zlim([0, 4])
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")

    plt.tight_layout()

    pos = {}
    for i in range(video_loader.get_number_of_frames()):
        graph = nx.Graph()
        frames = video_loader.get_frames()
        tracker.detect_and_track(frames)
        detections = tracker.get_detections()
        point_3d_list = []
        new_ids = []
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

            # Draw bounding boxes and IDs on frames
            cv2.putText(
                frame,
                "CLASS: " + str(int(name)),
                (int(bbox[0]), int(bbox[3])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 245, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                utils.id_to_rgb_color(id),
                2,
            )

            cv2.putText(
                frame,
                str("ID: " + str(int(id))),
                (int(centroid[0]), int(centroid[1] + 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.circle(
                frame,
                (int(centroid[0]), int(centroid[1])),
                5,
                utils.id_to_rgb_color(id),
                -1,
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
        for idx, c in enumerate(nx.connected_components(graph)):
            subgraph = graph.subgraph(c)
            if len(subgraph.nodes) > 1:
                ids = sorted(subgraph.nodes)
                d2_points = []
                proj_matricies = []

                for node in ids:
                    cam = int(node.split("cam")[1].split("id")[0])
                    id = int(node.split("id")[1])
                    centroid = subgraph.nodes[node]["centroid"]
                    bbox = subgraph.nodes[node]["bbox"]
                    P_cam = matcher.P_all[cam]
                    d2_points.append((bbox[2], bbox[3]))
                    proj_matricies.append(P_cam)

                if len(d2_points) >= 2:
                    point_3d = triangulate_point_from_multiple_views_linear(
                        proj_matricies, d2_points
                    )
                    print(f"3D point: {point_3d}")
                    point_3d_list.append(point_3d)
                    new_ids.append(idx)
                    
                    color = utils.id_to_rgb_color(idx)
                    
                    for node in subgraph.nodes:
                        node_color_map[node] = utils.normalize_rgb_color(color)
                        

        new_ids = [int(i) for i in range(len(point_3d_list))]
        # Create video mosaic with annotations
        processed_frames = []
        for frame in frames:
            # Convert BGR to RGB for matplotlib
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frames.append(cv2.resize(rgb_frame, (540, 360)))

        # Create video mosaic
        top_row = np.hstack((processed_frames[0], processed_frames[1]))
        bottom_row = np.hstack((processed_frames[2], processed_frames[3]))
        full_mosaic = np.vstack((top_row, bottom_row))
        video_img.set_data(full_mosaic)

        for node in graph.nodes():
            if node not in pos:
                cam = int(node.split("cam")[1].split("id")[0])
                obj_id = int(node.split("id")[1])
                x = (cam - 1) * 6  # Increased horizontal spacing
                y = obj_id * 3      # Increased vertical spacing
                pos[node] = (x, y)

                
        # Update graph visualization
        visualize_graph(graph, graph_ax, i, pos, node_color_map)

        # Update 3D plot
        ax_3d.clear()
        
        for point_idx, point_3d in enumerate(point_3d_list):
            color_rgb = utils.normalize_rgb_color(utils.id_to_rgb_color(point_idx))
            ax_3d.scatter(
                point_3d[0],
                point_3d[1],
                point_3d[2],
                c=[color_rgb],
                s=50,
                label=f"ID: {point_idx}",
            )
        ax_3d.set_xlim([-4, 4])
        ax_3d.set_ylim([-4, 4])
        ax_3d.set_zlim([0, 4])
        ax_3d.legend(loc="upper right")

        # Update figure and handle events
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Small delay for smooth visualization
        plt.pause(0.1)
        final_frame = utils.fig_to_image(fig)

        if i == 0:
            video = cv2.VideoWriter(
                "output.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (final_frame.shape[1], final_frame.shape[0]),
            )

        final_frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        video.write(final_frame_rgb)

        # Check for exit condition
        if not plt.fignum_exists(fig.number):
            break

    # Cleanup
    plt.ioff()
    plt.close()
    video_loader.release()


if __name__ == "__main__":
    main()
