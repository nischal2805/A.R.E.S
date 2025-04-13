from ultralytics import YOLO
import numpy as np
from config.settings import CONFIDENCE_THRESHOLD

class YOLODetector:
    def __init__(self, model_path=None):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model
        """
        from config.settings import YOLO_MODEL
        
        # Load YOLO model
        self.model = YOLO(model_path if model_path else YOLO_MODEL)
        
        # Enable model optimization if available
        if hasattr(self.model, 'to'):
            # Try to use GPU if available
            try:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    print("Using CUDA acceleration")
            except:
                pass
    
    def detect_humans(self, frame):
        """
        Detect humans in the given frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of bounding boxes for detected humans [x1, y1, x2, y2, confidence]
        """
        # Run YOLO detection
        results = self.model(frame)
        
        # Extract human detections (class 0 in COCO dataset)
        humans = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if the detected object is a person (class 0)
                if box.cls == 0:  # Person class
                    confidence = float(box.conf)
                    if confidence >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        humans.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence
                        })
        
        return humans