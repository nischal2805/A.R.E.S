from ultralytics import YOLO
import numpy as np
import torch
import cv2
import os
from huggingface_hub import hf_hub_download

class YOLODetector:
    def __init__(self, model_path=None):
        """Initialize with specialized face detection model"""
        # Define paths for model downloads
        model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(model_folder, exist_ok=True)
        
        # Path to save the downloaded model
        model_file = os.path.join(model_folder, "yolov8_face_detection.pt")
        
        # Check if model exists, download if not
        if not os.path.exists(model_file):
            print("Downloading face detection model from Hugging Face...")
            try:
                # Download model from Hugging Face
                model_file = hf_hub_download(
                    repo_id="arnabdhar/YOLOv8-Face-Detection",
                    filename="model.pt",
                    local_dir=model_folder,
                    local_dir_use_symlinks=False
                )
                # Rename to our standard name
                os.rename(model_file, os.path.join(model_folder, "yolov8_face_detection.pt"))
                model_file = os.path.join(model_folder, "yolov8_face_detection.pt")
                print(f"Model downloaded to {model_file}")
            except Exception as e:
                print(f"Error downloading model: {e}")
                # Fall back to standard YOLO model if face model download fails
                model_file = model_path
                print(f"Falling back to standard model: {model_path}")
        
        # Load the model with optimizations
        print(f"Loading model from: {model_file}")
        self.face_model = YOLO(model_file)
        
        # Set model parameters for speed
        self.face_model.fuse()  # Fuse model layers for speed
        
        # Try GPU acceleration with half precision
        if torch.cuda.is_available():
            self.face_model.to('cuda')
            # Use half precision on CUDA for 2x faster inference
            self.face_model.model.half()  
            print(f"Using GPU with half precision: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for detection")
        
        print("Face detection model initialized")
            
    def detect_humans(self, frame):
        """Detect faces using YOLOv8 face detection model with speed optimizations"""
        # Resize frame for faster processing  
        resized_frame = cv2.resize(frame, (320, 240))
        
        # Run face detection with optimized parameters
        results = self.face_model(resized_frame, conf=0.4, iou=0.5, verbose=False)
        
        # Scale bounding boxes back to original frame size
        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240
        
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                # Scale coordinates back to original image
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                conf = float(box.conf)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'id': None,
                })
        
        return detections