import cv2
import numpy as np
import mediapipe as mp

class EyeTracker:
    def __init__(self, offset_y=20, smoothing=0.5):
        """
        Initialize eye tracker
        
        Args:
            offset_y: Vertical offset to find forehead center
            smoothing: Smoothing factor for eye tracking (0-1)
        """
        self.offset_y = offset_y
        self.smoothing = smoothing
        self.previous_targets = {}
        
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def find_targets(self, frame, detections):
        """
        Find forehead targets for each detected person
        
        Args:
            frame: Input image frame
            detections: List of human detections from YOLO
            
        Returns:
            List of target points (x, y) for each detection
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        targets = []
        current_ids = set()
        
        # Process each human detection
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Extract face region with some margin
            face_region = rgb_frame[max(0, y1-20):min(frame.shape[0], y2+20), 
                                  max(0, x1-20):min(frame.shape[1], x2+20)]
            
            if face_region.size == 0:
                continue
                
            # Process with MediaPipe
            results = self.face_mesh.process(face_region)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get eye landmarks (left eye: 362, right eye: 133)
                    left_eye = face_landmarks.landmark[362]
                    right_eye = face_landmarks.landmark[133]
                    
                    # Calculate eye midpoint in relative coordinates
                    eye_mid_x = (left_eye.x + right_eye.x) / 2
                    eye_mid_y = (left_eye.y + right_eye.y) / 2
                    
                    # Convert to absolute coordinates in the original frame
                    region_height, region_width = face_region.shape[:2]
                    abs_x = int(x1 - 20 + eye_mid_x * region_width)
                    abs_y = int(y1 - 20 + eye_mid_y * region_height)
                    
                    # Apply vertical offset to target forehead
                    target_y = max(0, abs_y - self.offset_y)
                    
                    # Apply smoothing if we have previous targets
                    person_id = f"person_{i}"
                    current_ids.add(person_id)
                    
                    if person_id in self.previous_targets:
                        prev_x, prev_y = self.previous_targets[person_id]
                        smoothed_x = int(prev_x * self.smoothing + abs_x * (1 - self.smoothing))
                        smoothed_y = int(prev_y * self.smoothing + target_y * (1 - self.smoothing))
                        target = (smoothed_x, smoothed_y)
                    else:
                        target = (abs_x, target_y)
                    
                    self.previous_targets[person_id] = target
                    targets.append({
                        'bbox': bbox,
                        'target': target,
                        'confidence': detection['confidence']
                    })
        
        # Remove tracking data for persons no longer in frame
        for pid in list(self.previous_targets.keys()):
            if pid not in current_ids:
                del self.previous_targets[pid]
        
        return targets