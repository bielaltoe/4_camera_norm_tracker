"""
This file contains the configuration for the four view tracker.
"""

CLASS_NAMES = {
0: 'person',
1: 'helmet',
25: 'umbrella',
56: 'chair',
 }

YOLO_MODEL = "models/yolo11x.pt"  # YOLO model file
# CLASSES = [i for i in range(79)]  # Classes to be detected
CONFIDENCE = 0.4  # Confidence threshold of the YOLO tracker model
DISTANCE_THRESHOLD = 0.2  # Distance threshold for the epipolar line matching
DRIFT_THRESHOLD = 0.4  # Drift threshold for the epipolar line matching quanto maior mais restritivo
