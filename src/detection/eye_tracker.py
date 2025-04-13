import cv2
import numpy as np

class EyeTracker:
    def __init__(self, offset_y=20, smoothing=0.5):
        """
        Initialize forehead target finder
        
        Args:
            offset_y: Vertical offset for fine-tuning (can be 0 with the new method)
            smoothing: Smoothing factor for tracking (0-1)
        """
        self.offset_y = offset_y
        self.smoothing = smoothing
        self.previous_targets = {}
    
    def find_targets(self, frame, detections):
        """
        Find forehead targets using bounding box method
        
        Args:
            frame: Input image frame
            detections: List of human detections from YOLO/tracker
            
        Returns:
            List of target points with metadata
        """
        targets = []
        current_ids = set()
        
        # Process each human detection
        for detection in detections:
            # Get the ID for tracking
            person_id = f"person_{detection.get('id', hash(str(detection['bbox'])))}"
            current_ids.add(person_id)
            
            # Extract coordinates from bounding box
            x1, y1, x2, y2 = detection['bbox']
            
            # Calculate forehead position (30% down from top of head)
            # Horizontally centered, 30% down from the top of the bounding box
            forehead_x = int((x1 + x2) / 2)
            forehead_y = int(y1 + (y2 - y1) * 0.30) - self.offset_y  # Subtract offset for fine-tuning
            
            # Apply smoothing if we have previous targets
            if person_id in self.previous_targets:
                prev_x, prev_y = self.previous_targets[person_id]
                smoothed_x = int(prev_x * self.smoothing + forehead_x * (1 - self.smoothing))
                smoothed_y = int(prev_y * self.smoothing + forehead_y * (1 - self.smoothing))
                target_point = (smoothed_x, smoothed_y)
            else:
                target_point = (forehead_x, forehead_y)
            
            # Update previous targets
            self.previous_targets[person_id] = target_point
            
            # Create target data
            target_data = {
                'bbox': detection['bbox'],
                'target': target_point,
                'confidence': detection['confidence'],
                'id': detection.get('id', None),
                'locked': detection.get('locked', False)
            }
            
            targets.append(target_data)
        
        # Remove tracking data for persons no longer in frame
        for pid in list(self.previous_targets.keys()):
            if pid not in current_ids:
                del self.previous_targets[pid]
        
        return targets