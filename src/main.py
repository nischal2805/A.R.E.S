import cv2
import time
from camera.threaded_camera import ThreadedCamera  # Use threaded camera
from detection.yolo_detector import YOLODetector
from detection.eye_tracker import EyeTracker
from visualization.overlay import TargetingOverlay
from config.settings import *

def main():
    # Initialize components
    camera = ThreadedCamera(CAMERA_IP, FRAME_WIDTH, FRAME_HEIGHT)
    camera.start()
    
    detector = YOLODetector(YOLO_MODEL)
    eye_tracker = EyeTracker(EYE_OFFSET_Y, EYE_TRACKING_SMOOTHING)
    overlay = TargetingOverlay(TARGET_CIRCLE_RADIUS, OVERLAY_CIRCLE_RADIUS, 
                              TARGET_COLOR, OVERLAY_COLOR)
    
    # Performance tracking
    prev_time = 0
    frame_count = 0
    
    while True:
        # Capture frame from camera
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)  # Sleep briefly to prevent CPU overload
            continue
        
        # Skip frames for better performance
        frame_count += 1
        if frame_count % (SKIP_FRAMES + 1) != 0:
            # Still display frame but skip detection
            cv2.imshow("Targeting System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Detect humans in the frame
        detections = detector.detect_humans(frame)
        
        # Track eyes and find forehead targets for each detection
        targets = eye_tracker.find_targets(frame, detections)
        
        # Apply targeting overlay
        result_frame = overlay.draw_targeting(frame, targets)
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        cv2.putText(result_frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow("Targeting System", result_frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()