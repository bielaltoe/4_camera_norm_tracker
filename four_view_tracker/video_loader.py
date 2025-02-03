"""
Video Loading and Synchronization Module

This module provides functionality for loading and synchronizing multiple video streams
simultaneously. It handles video capture operations and ensures synchronized frame retrieval
across all cameras.
"""

import cv2
import logging


class VideoLoader:
    """
    A class to handle multiple synchronized video streams.
    
    This class manages multiple video captures simultaneously, providing synchronized
    access to frames from all cameras. It ensures that frame retrieval is properly
    coordinated across all video sources.
    
    Attributes:
        sources_list (list): List of video source paths
        video_captures (list): List of OpenCV VideoCapture objects
    """

    def __init__(self, sources_list: list):
        """
        Initialize VideoLoader with multiple video sources.
        
        Args:
            sources_list (list): List of paths to video files or camera indices
        
        Raises:
            RuntimeError: If any video source fails to open
        """
        self.sources_list = sources_list
        self.video_captures = []
        
        # Initialize video captures for each source
        for source in self.sources_list:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {source}")
            self.video_captures.append(cap)

        logging.info(f"VideoLoader initialized with {len(sources_list)} sources")

    def get_frames(self):
        """
        Retrieve synchronized frames from all video sources.
        
        Returns:
            list: List of frames from all cameras, in the same order as sources_list.
                 Returns None for any failed frame reads.
        """
        frames = []
        for idx, video_capture in enumerate(self.video_captures):
            ret, frame = video_capture.read()
            if not ret:
                logging.warning(f"Failed to read frame from source {self.sources_list[idx]}")
                frames.append(None)
            else:
                frames.append(frame)
        return frames

    def get_number_of_frames(self):
        """
        Get the minimum number of frames across all video sources.
        
        This ensures synchronization by using the shortest video length
        when videos have different durations.
        
        Returns:
            int: Minimum number of frames across all video sources
        """
        num_frames = []
        for idx, video_capture in enumerate(self.video_captures):
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            logging.info(f"Source {self.sources_list[idx]}: {frame_count} frames")
            num_frames.append(frame_count)
        return min(num_frames)

    def release(self):
        """
        Release all video captures and free resources.
        
        Should be called when done with video processing to properly
        clean up resources.
        """
        for video_capture in self.video_captures:
            video_capture.release()
        logging.info("Released all video captures")