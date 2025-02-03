"""
Detection Matching Module

This module provides functionality for matching object detections across multiple camera views
using epipolar geometry. It manages fundamental matrices and projection matrices for all
camera pairs and implements matching logic based on epipolar constraints.
"""

import logging
import numpy as np
import cv2
from ploting_utils import Utils
from load_fundamental_matrices import FundamentalMatrices
from config import DISTANCE_THRESHOLD
from epipolar_utils import calculate_lines, cross_distance, dist_p_l


class Matcher:
    """
    A class to match object detections across multiple camera views.
    
    This class handles the matching of detections between different camera views using
    epipolar geometry constraints. It maintains the fundamental and projection matrices
    for all camera pairs and provides methods for visualization and matching.
    
    Attributes:
        F_all (dict): Dictionary of fundamental matrices for each camera pair
        P_all (dict): Dictionary of projection matrices for each camera
        lines_1_2 (np.ndarray): Epipolar lines from camera 1 to camera 2
        lines_2_1 (np.ndarray): Epipolar lines from camera 2 to camera 1
        colors (dict): Color mapping for visualization
    """

    def __init__(self):
        """
        Initialize the Matcher with camera configuration files.
        
        Loads fundamental and projection matrices from camera configuration files.
        Currently supports a fixed 4-camera setup.
        """
        logging.info("Initializing Matcher")
        
        # Load camera configurations
        camera_configs = [
            "config_camera/0.json",
            "config_camera/1.json",
            "config_camera/2.json",
            "config_camera/3.json",
        ]
        
        # Initialize fundamental and projection matrices
        matrices = FundamentalMatrices()
        self.F_all = matrices.fundamental_matrices_all(camera_configs)
        self.P_all = matrices.projection_matrices_all(camera_configs)

        # Initialize storage for epipolar lines
        self.lines_1_2 = None
        self.lines_2_1 = None
        self.colors = {}

    def plot_lines(self, img1, img2):
        """
        Plots epipolar lines on image pairs for visualization.
        
        Args:
            img1 (np.ndarray): First camera image
            img2 (np.ndarray): Second camera image
            
        Returns:
            tuple: Pair of images with epipolar lines drawn
        """
        # Draw lines on second image
        for r in self.lines_1_2:
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])
            img2 = cv2.line(img2, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Draw lines on first image
        for r2 in self.lines_2_1:
            x0_2, y0_2 = map(int, [0, -r2[2] / r2[1]])
            x1_2, y1_2 = map(int, [img1.shape[1], -(r2[2] + r2[0] * img1.shape[1]) / r2[1]])
            img1 = cv2.line(img1, (x0_2, y0_2), (x1_2, y1_2), (0, 255, 0), 2)

        return img1, img2

    def match_detections(self, detections, cams):
        """
        Match detections between two camera views using epipolar geometry.
        
        This method implements a sophisticated matching algorithm that:
        1. Computes epipolar lines between camera pairs
        2. Calculates cross-distances between detections
        3. Filters matches based on distance threshold and uniqueness
        4. Handles ambiguous matches using a drift threshold
        
        Args:
            detections (list): List of ObjectDetection instances
            cams (list): List of two camera indices to match between
            
        Returns:
            list: List of matched detection pairs that satisfy the epipolar constraints
        """
        # Initialize epipolar lines
        self.lines_1_2 = []
        self.lines_2_1 = []

        # Get fundamental matrices for the camera pair
        F_1_2 = self.F_all[cams[0]][cams[1]]
        F_2_1 = self.F_all[cams[1]][cams[0]]

        # Filter detections by camera
        detections_cam_1 = [det for det in detections if det.cam == cams[0]]
        detections_cam_2 = [det for det in detections if det.cam == cams[1]]

        if len(detections_cam_1) < 1 or len(detections_cam_2) < 1:
            return []

        # Extract centroids and calculate epipolar lines
        centroids_cam_1 = np.array([det.centroid for det in detections_cam_1])
        centroids_cam_2 = np.array([det.centroid for det in detections_cam_2])
        
        self.lines_1_2 = calculate_lines(F_1_2, centroids_cam_1)
        self.lines_2_1 = calculate_lines(F_2_1, centroids_cam_2)

        # Find potential matches based on epipolar geometry
        maybe_matches = []
        for i, det_cam_1 in enumerate(detections_cam_1):
            for j, det_cam_2 in enumerate(detections_cam_2):
                d_cross = cross_distance(
                    det_cam_1.bbox, det_cam_2.bbox, self.lines_1_2[i], self.lines_2_1[j]
                )
                if det_cam_1.name == det_cam_2.name:
                    maybe_matches.append([det_cam_1, det_cam_2, d_cross])

        # Sort matches by cross distance and filter based on threshold
        sorted_maybe_matches = sorted(maybe_matches, key=lambda x: x[2])
        filtered_list = []

        # First pass: filter based on distance threshold
        for maybe_match in sorted_maybe_matches:
            if maybe_match[2] < DISTANCE_THRESHOLD:
                if len(filtered_list) == 0:
                    filtered_list.append(maybe_match)
                else:
                    for f in filtered_list:
                        if maybe_match[0].id != f[0].id and maybe_match[1].id != f[1].id:
                            filtered_list.append(maybe_match)
                            break

        # Second pass: handle ambiguous matches
        final_list = filtered_list.copy()
        drift = 0.05  # Threshold for considering matches as ambiguous

        for filtered_match in filtered_list:
            for maybe_match in sorted_maybe_matches:
                if ((maybe_match[0].id != filtered_match[0].id and 
                     maybe_match[1].id == filtered_match[1].id) or
                    (maybe_match[0].id == filtered_match[0].id and 
                     maybe_match[1].id != filtered_match[1].id)):
                    
                    # Remove ambiguous matches that are too close in score
                    if abs(maybe_match[2] - filtered_match[2]) < drift:
                        if filtered_match in final_list:
                            final_list.remove(filtered_match)

        # Log successful matches
        for match in final_list:
            det_cam_1, det_cam_2, d_cross = match
            logging.info(f"\nMatch found between cameras {cams[0]} and {cams[1]}:")
            logging.info(f"Camera {cams[0]} Detection - ID: {det_cam_1.id}, "
                        f"Class: {int(det_cam_1.name)}, "
                        f"Centroid: ({det_cam_1.centroid[0]:.2f}, {det_cam_1.centroid[1]:.2f})")
            logging.info(f"Camera {cams[1]} Detection - ID: {det_cam_2.id}, "
                        f"Class: {int(det_cam_2.name)}, "
                        f"Centroid: ({det_cam_2.centroid[0]:.2f}, {det_cam_2.centroid[1]:.2f})")
            logging.info(f"Cross Distance: {d_cross:.4f}")

        return final_list
