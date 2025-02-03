YOLO_MODEL = "yolo11x.pt"  # YOLO model file
NUM_CAM = 3  # Number of cameras
# CLASSES = [i for i in range(79)]  # Classes to be detected
CONFIDENCE = 0.85  # Confidence threshold of the YOLO tracker model
DISTANCE_THRESHOLD = 0.3  # Distance threshold for the epipolar line matching
DISTANCE_RATIO_THRESHOLD = (
    2.5  # Distance ratio threshold for the epipolar line matching
)
CLASSES = [0]