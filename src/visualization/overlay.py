import cv2
import numpy as np

class TargetingOverlay:
    def __init__(self, target_radius=10, overlay_radius=30, 
                 target_color=(255, 0, 0), overlay_color=(0, 255, 0)):
        """
        Initialize targeting overlay
        
        Args:
            target_radius: Radius of the target circle
            overlay_radius: Radius of the overlay circle
            target_color: Color for the target circle (BGR)
            overlay_color: Color for the overlay circle (BGR)
        """
        self.target_radius = target_radius
        self.overlay_radius = overlay_radius
        self.target_color = target_color
        self.overlay_color = overlay_color
    
    def draw_targeting(self, frame, targets):
        """
        Draw targeting overlay on the frame
        
        Args:
            frame: Input image frame
            targets: List of target points with bounding boxes
            
        Returns:
            Frame with targeting overlay
        """
        result_frame = frame.copy()
        
        for target_info in targets:
            # Draw bounding box
            x1, y1, x2, y2 = target_info['bbox']
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), self.overlay_color, 2)
            
            # Draw target circles
            target_x, target_y = target_info['target']
            
            # Draw larger hollow circle
            cv2.circle(result_frame, (target_x, target_y), 
                      self.overlay_radius, self.overlay_color, 2)
            
            # Draw smaller filled circle for the target
            cv2.circle(result_frame, (target_x, target_y), 
                      self.target_radius, self.target_color, -1)
            
            # Draw crosshair
            cv2.line(result_frame, (target_x - 15, target_y), 
                    (target_x + 15, target_y), self.target_color, 1)
            cv2.line(result_frame, (target_x, target_y - 15), 
                    (target_x, target_y + 15), self.target_color, 1)
            
            # Show confidence
            confidence = target_info['confidence']
            cv2.putText(result_frame, f"{confidence:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.overlay_color, 1)
        
        return result_frame