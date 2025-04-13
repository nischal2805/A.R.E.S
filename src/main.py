import cv2
import time
import numpy as np
from detection.yolo_detector import YOLODetector
from detection.tracker import PersonTracker
from detection.eye_tracker import EyeTracker
from visualization.overlay import TargetingOverlay
from config.settings import *

def main():
    detector = YOLODetector(YOLO_MODEL)
    tracker = PersonTracker(max_disappeared=10)  # New tracker component
    eye_tracker = EyeTracker(EYE_OFFSET_Y, EYE_TRACKING_SMOOTHING)
    overlay = TargetingOverlay(TARGET_CIRCLE_RADIUS, OVERLAY_CIRCLE_RADIUS, 
                              TARGET_COLOR, OVERLAY_COLOR)
    
    # Open camera
    camera = cv2.VideoCapture(CAMERA_IP)
    if not camera.isOpened():
        print("Error: Could not open camera. Check your camera IP and connection.")
        return
    
    # Performance tracking
    prev_time = time.time()
    detection_interval = SKIP_FRAMES  # Run detection every N frames
    frame_count = 0
    
    while True:
        # Capture frame
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame")
            time.sleep(0.5)
            continue
        
        # Resize for better performance
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Only run detection every few frames to improve performance
        run_detection = frame_count % detection_interval == 0
        
        if run_detection:
            # Detect humans
            detected_humans = detector.detect_humans(frame)
            
            # Update tracker with new detections
            tracked_humans = tracker.update(frame, detected_humans)
            
            # Track eyes and find targets
            targets = eye_tracker.find_targets(frame, tracked_humans)
        else:
            # Just update tracker positions without new detections
            tracked_humans = tracker.update(frame, [])
            
            # Update targets based on tracking only
            targets = eye_tracker.find_targets(frame, tracked_humans)
        
        # Create visual overlay
        result = overlay.draw_targeting(frame, targets)
        
        # Show lock-on status
        for target in targets:
            if 'locked' in target and target['locked']:
                # Draw lock-on indicator
                x, y = target['target']
                cv2.putText(result, "LOCKED", (x-30, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Display FPS
        cv2.putText(result, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display result
        cv2.imshow("Targeting System", result)
        
        # Update frame counter
        frame_count += 1
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()