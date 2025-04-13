import cv2
import numpy as np

class PersonTracker:
    def __init__(self, max_disappeared=30):
        """Tracking system that locks onto detected persons"""
        self.next_id = 0
        self.trackers = {}  # Stores active trackers
        self.disappeared = {}  # Tracks frames since last seen
        self.max_disappeared = max_disappeared
        
    def update(self, frame, detections):
        """Update tracking with new detections and lock onto targets"""
        if len(detections) == 0:
            # Mark all existing trackers as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Remove if disappeared for too many frames
                if self.disappeared[object_id] > self.max_disappeared:
                    self.delete_tracker(object_id)
            
            return self.get_tracked_objects()
        
        # Initialize trackers for new detections
        if len(self.trackers) == 0:
            for detection in detections:
                self.register(frame, detection)
        else:
            # Get current object IDs
            object_ids = list(self.trackers.keys())
            
            # Update existing trackers
            for object_id in object_ids:
                # Get tracker
                tracker = self.trackers[object_id]['tracker']
                
                # Update tracker
                success, bbox = tracker.update(frame)
                
                if success:
                    # Convert to proper format
                    x, y, w, h = [int(v) for v in bbox]
                    self.trackers[object_id]['bbox'] = [x, y, x+w, y+h]
                    self.disappeared[object_id] = 0
                else:
                    self.disappeared[object_id] += 1
                    
                    # Remove if disappeared for too many frames
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.delete_tracker(object_id)
            
            # Match new detections with existing trackers using IoU
            used_trackers = set()
            used_detections = set()
            
            for i, detection in enumerate(detections):
                best_iou = 0.3  # Minimum IoU threshold
                best_id = None
                
                for object_id in self.trackers:
                    if object_id in used_trackers:
                        continue
                    
                    # Calculate IoU between detection and tracker
                    iou = self.calculate_iou(detection['bbox'], self.trackers[object_id]['bbox'])
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_id = object_id
                
                # Match found - update existing tracker
                if best_id is not None:
                    self.update_tracker(frame, best_id, detection)
                    used_trackers.add(best_id)
                    used_detections.add(i)
            
            # Register new detections that weren't matched
            for i, detection in enumerate(detections):
                if i not in used_detections:
                    self.register(frame, detection)
        
        return self.get_tracked_objects()
    
    def register(self, frame, detection):
        """Register new tracker for a detection"""
        # Create a new tracker
        tracker = cv2.TrackerKCF_create()
        
        # Initialize tracker
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        tracker.init(frame, (x1, y1, x2-x1, y2-y1))
        
        # Store tracker with original data
        self.trackers[self.next_id] = {
            'tracker': tracker,
            'bbox': bbox,
            'confidence': detection['confidence'],
            'lock_status': False
        }
        
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def update_tracker(self, frame, object_id, detection):
        """Update existing tracker with new detection"""
        # Update tracker
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Re-initialize tracker with new detection
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, (x1, y1, x2-x1, y2-y1))
        
        # Update tracker data
        self.trackers[object_id]['tracker'] = tracker
        self.trackers[object_id]['bbox'] = bbox
        self.trackers[object_id]['confidence'] = detection['confidence']
        self.trackers[object_id]['lock_status'] = True  # Mark as locked-on
        self.disappeared[object_id] = 0
    
    def delete_tracker(self, object_id):
        """Delete a tracker"""
        del self.trackers[object_id]
        del self.disappeared[object_id]
    
    def get_tracked_objects(self):
        """Get currently tracked objects"""
        objects = []
        for object_id, data in self.trackers.items():
            objects.append({
                'id': object_id,
                'bbox': data['bbox'],
                'confidence': data['confidence'],
                'locked': data['lock_status']
            })
        return objects
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas of both bounding boxes
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate IoU
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return iou