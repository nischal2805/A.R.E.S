# filepath: yolo-targeting-system/yolo-targeting-system/src/config/settings.py

# Configuration settings for the YOLO targeting system

# Model settings
YOLO_MODEL = "yolov8n.pt"  # Use smallest YOLOv8 model for performance

# Detection parameters
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
SKIP_FRAMES = 2  # Process every 3rd frame for better performance

# Eye tracking parameters
EYE_OFFSET_Y = 20  # Vertical offset to find forehead center
EYE_TRACKING_SMOOTHING = 0.5  # Smoothing factor for eye tracking

# Camera settings
CAMERA_IP = "http://192.168.1.23:8080/video"
FRAME_WIDTH = 640  # Reduced for better performance
FRAME_HEIGHT = 380 # Reduced for better performance
FPS = 30  

# Visualization settings
TARGET_CIRCLE_RADIUS = 8  # Radius of the target circle
OVERLAY_CIRCLE_RADIUS = 25  # Radius of the overlay circle
OVERLAY_COLOR = (0, 255, 0)  # Color for the overlay (Green in BGR)
TARGET_COLOR = (255, 0, 0)  # Color for the target (Red in BGR)