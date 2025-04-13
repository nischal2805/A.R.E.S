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
        """Draw targeting overlay with lock-on visuals"""
        result_frame = frame.copy()
        
        for target_info in targets:
            # Get bounding box
            x1, y1, x2, y2 = target_info['bbox']
            
            # Determine color based on lock status
            color = (0, 0, 255) if target_info.get('locked', False) else self.overlay_color
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw target circles
            target_x, target_y = target_info['target']
            
            # Draw larger targeting circle
            cv2.circle(result_frame, (target_x, target_y), 
                      self.overlay_radius, color, 2)
            
            # Draw smaller aiming point
            cv2.circle(result_frame, (target_x, target_y), 
                      self.target_radius, self.target_color, -1)
            
            # Add dynamic crosshair
            size = self.overlay_radius + 5 if target_info.get('locked', False) else 15
            cv2.line(result_frame, (target_x - size, target_y), 
                    (target_x + size, target_y), self.target_color, 2)
            cv2.line(result_frame, (target_x, target_y - size), 
                    (target_x, target_y + size), self.target_color, 2)
            
            # Show confidence and ID
            label = f"ID:{target_info.get('id', '?')} {target_info['confidence']:.2f}"
            cv2.putText(result_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add lock-on indicator for locked targets
            if target_info.get('locked', False):
                # Draw lock corners
                corner_len = 20
                # Top-left
                cv2.line(result_frame, (x1, y1), (x1 + corner_len, y1), (0, 0, 255), 2)
                cv2.line(result_frame, (x1, y1), (x1, y1 + corner_len), (0, 0, 255), 2)
                # Top-right
                cv2.line(result_frame, (x2, y1), (x2 - corner_len, y1), (0, 0, 255), 2)
                cv2.line(result_frame, (x2, y1), (x2, y1 + corner_len), (0, 0, 255), 2)
                # Bottom-left
                cv2.line(result_frame, (x1, y2), (x1 + corner_len, y2), (0, 0, 255), 2)
                cv2.line(result_frame, (x1, y2), (x1, y2 - corner_len), (0, 0, 255), 2)
                # Bottom-right
                cv2.line(result_frame, (x2, y2), (x2 - corner_len, y2), (0, 0, 255), 2)
                cv2.line(result_frame, (x2, y2), (x2, y2 - corner_len), (0, 0, 255), 2)
        
        return result_frame