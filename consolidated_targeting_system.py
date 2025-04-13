import cv2
import os
import numpy as np
import time
import threading
import queue
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import face_recognition

# ============= CONFIGURATION =============
# Camera/Video settings
CAMERA_IP = "http://192.168.1.23:8080/video"  # IP camera URL (can be replaced with video path)
USE_VIDEO_FILE = False  # Set to True to use video file instead of IP camera
VIDEO_FILE_PATH = ""  # Path to video file if USE_VIDEO_FILE is True
FRAME_WIDTH = 640  # Frame width
FRAME_HEIGHT = 480  # Frame height

# YOLO model settings
USE_FACE_MODEL = True  # Use specialized face detection model
FACE_MODEL_REPO = "arnabdhar/YOLOv8-Face-Detection"
FACE_MODEL_FILENAME = "model.pt"
FALLBACK_MODEL = "yolov8n.pt"  # Fallback to standard YOLOv8 model if face model fails

# Detection parameters
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5
SKIP_FRAMES = 1  # Process every N+1 frames (0 = process all)
MAX_DISAPPEARED = 3  # Frames before deleting a tracker (reduced for less ghost tracking)

# Face recognition settings
ENABLE_FACE_RECOGNITION = True
KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")
FACE_RECOGNITION_TOLERANCE = 0.45  # Lower = stricter matching

# Tracking parameters
FOREHEAD_OFFSET_RATIO = 0.30  # Percentage from top of detection box
VERTICAL_OFFSET = 20  # Additional vertical offset (pixels)
SMOOTHING_FACTOR = 0.7  # Higher = smoother tracking (0-1)

# Visualization settings
TARGET_CIRCLE_RADIUS = 8
OVERLAY_CIRCLE_RADIUS = 25
TARGET_COLOR = (255, 0, 0)  # Blue (BGR)
OVERLAY_COLOR = (0, 255, 0)  # Green (BGR)
LOCKED_COLOR = (0, 0, 255)  # Red (BGR)
SHOW_FPS = True
SHOW_BOUNDING_BOXES = True

# ============= FACE RECOGNITION =============
class FaceRecognizer:
    def __init__(self, known_faces_dir=KNOWN_FACES_DIR, tolerance=FACE_RECOGNITION_TOLERANCE):
        self.known_face_encodings = []
        self.known_face_names = []
        self.tolerance = tolerance
        self.locked_encoding = None
        self.locked_name = None
        
        # Create known_faces directory if it doesn't exist
        os.makedirs(known_faces_dir, exist_ok=True)
        
        # Load known faces
        if os.path.exists(known_faces_dir):
            print(f"Loading known faces from {known_faces_dir}")
            for filename in os.listdir(known_faces_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        path = os.path.join(known_faces_dir, filename)
                        img = face_recognition.load_image_file(path)
                        encodings = face_recognition.face_encodings(img)
                        if encodings:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(os.path.splitext(filename)[0])
                    except Exception as e:
                        print(f"Error loading face {filename}: {e}")
            
            print(f"Loaded {len(self.known_face_names)} known faces.")
    
    def recognize_face(self, face_img):
        """Recognize a face in the given image region"""
        # Convert to RGB (face_recognition uses RGB)
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Get face encodings
        encodings = face_recognition.face_encodings(rgb_face)
        
        if not encodings:
            return "Unknown", False
        
        face_encoding = encodings[0]
        
        # Check if we're tracking a specific face
        if self.locked_encoding is not None:
            match = face_recognition.compare_faces([self.locked_encoding], face_encoding, 
                                                 tolerance=self.tolerance)
            if match[0]:
                return self.locked_name, True
            return "Unknown", False
            
        # Compare with known faces
        if self.known_face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, 
                                                  tolerance=self.tolerance)
            
            if True in matches:
                # Find the best match
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = self.known_face_names[best_match_index]
                
                # Lock onto this face
                self.locked_encoding = face_encoding
                self.locked_name = name
                return name, True
        
        return "Unknown", False
    
    def clear_lock(self):
        """Clear the locked face tracking"""
        self.locked_encoding = None
        self.locked_name = None

# ============= PERSON TRACKER =============
class PersonTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED):
        self.next_id = 0
        self.trackers = {}  # Stores active trackers
        self.disappeared = {}  # Tracks frames since last seen
        self.max_disappeared = max_disappeared
        
    def update(self, frame, detections):
        """Update tracking with new detections"""
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
            'face_name': detection.get('face_name', 'Unknown'),
            'lock_status': detection.get('lock_status', False)
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
        
        # Update face name if available
        if 'face_name' in detection:
            self.trackers[object_id]['face_name'] = detection['face_name']
            
        # Update lock status if available
        if 'lock_status' in detection:
            self.trackers[object_id]['lock_status'] = detection['lock_status']
            
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
                'face_name': data.get('face_name', 'Unknown'),
                'locked': data.get('lock_status', False)
            })
        return objects
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes - optimized version"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Fast rejection test
        if x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2:
            return 0.0
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate areas of both bounding boxes
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate IoU
        return intersection_area / float(bbox1_area + bbox2_area - intersection_area)

# ============= TARGET TRACKER =============
class TargetTracker:
    def __init__(self, offset_y=VERTICAL_OFFSET, smoothing=SMOOTHING_FACTOR):
        self.offset_y = offset_y
        self.smoothing = smoothing
        self.previous_targets = {}
    
    def find_targets(self, frame, detections):
        """Find forehead targets"""
        targets = []
        current_ids = set()
        
        # Process each detected face
        for detection in detections:
            # Get the ID for tracking
            person_id = f"person_{detection.get('id', hash(str(detection['bbox'])))}"
            current_ids.add(person_id)
            
            # Extract coordinates from bounding box
            x1, y1, x2, y2 = detection['bbox']
            
            # Calculate forehead position
            forehead_x = int((x1 + x2) / 2)
            forehead_y = int(y1 + (y2 - y1) * FOREHEAD_OFFSET_RATIO) - self.offset_y
            
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
                'locked': detection.get('locked', False),
                'face_name': detection.get('face_name', 'Unknown')
            }
            
            targets.append(target_data)
        
        # Remove tracking data for persons no longer in frame
        for pid in list(self.previous_targets.keys()):
            if pid not in current_ids:
                del self.previous_targets[pid]
        
        return targets

# ============= YOLO DETECTOR =============
class FaceDetector:
    def __init__(self):
        """Initialize face detection model"""
        # Determine model file path
        if USE_FACE_MODEL:
            try:
                # Download from Hugging Face if needed
                print("Using specialized face detection model from Hugging Face")
                model_file = hf_hub_download(
                    repo_id=FACE_MODEL_REPO,
                    filename=FACE_MODEL_FILENAME,
                    local_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
                    local_dir_use_symlinks=False
                )
                print(f"Model loaded from: {model_file}")
            except Exception as e:
                print(f"Error using face detection model: {e}")
                print(f"Falling back to standard YOLO model: {FALLBACK_MODEL}")
                model_file = FALLBACK_MODEL
        else:
            model_file = FALLBACK_MODEL
            print(f"Using standard YOLO model: {model_file}")
            
        # Load the model
        self.model = YOLO(model_file)
        
        # Set model parameters for performance
        self.model.fuse()
        
        # Use GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')
            # Use half precision on CUDA for faster inference
            self.model.model.half()
            print(f"Using GPU with half precision: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for detection")
    
    def detect(self, frame):
        """Detect faces in a frame"""
        # Resize for faster processing
        resized_frame = cv2.resize(frame, (320, 240))
        
        # Run detection
        results = self.model(resized_frame, conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD, verbose=False)
        
        # Scale bounding boxes back to original frame size
        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240
        
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Get class - in face model all boxes are faces, in standard YOLO check for person class (0)
                if USE_FACE_MODEL or int(box.cls[0]) == 0:  # class 0 is person in COCO
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    # Scale coordinates back to original image
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    # Make sure box is within frame bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    # Skip if box is too small
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue
                        
                    conf = float(box.conf)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'id': None,
                    })
        
        return detections

# ============= VISUALIZATION =============
def draw_targeting_overlay(frame, targets):
    """Draw targeting overlay with lock-on visuals"""
    result_frame = frame.copy()
    
    for target_info in targets:
        # Get bounding box
        x1, y1, x2, y2 = target_info['bbox']
        
        # Determine color based on lock status
        color = LOCKED_COLOR if target_info.get('locked', False) else OVERLAY_COLOR
        
        # Draw bounding box if enabled
        if SHOW_BOUNDING_BOXES:
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw target circles
        target_x, target_y = target_info['target']
        
        # Draw larger targeting circle
        cv2.circle(result_frame, (target_x, target_y), OVERLAY_CIRCLE_RADIUS, color, 2)
        
        # Draw smaller aiming point
        cv2.circle(result_frame, (target_x, target_y), TARGET_CIRCLE_RADIUS, TARGET_COLOR, -1)
        
        # Add dynamic crosshair
        size = OVERLAY_CIRCLE_RADIUS + 5 if target_info.get('locked', False) else 15
        cv2.line(result_frame, (target_x - size, target_y), 
                (target_x + size, target_y), TARGET_COLOR, 2)
        cv2.line(result_frame, (target_x, target_y - size), 
                (target_x, target_y + size), TARGET_COLOR, 2)
        
        # Show face name, confidence and ID
        name = target_info.get('face_name', 'Unknown')
        label = f"{name} ID:{target_info.get('id', '?')} {target_info['confidence']:.2f}"
        cv2.putText(result_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add lock-on indicator for locked targets
        if target_info.get('locked', False):
            # Draw lock corners
            corner_len = 20
            # Top-left
            cv2.line(result_frame, (x1, y1), (x1 + corner_len, y1), LOCKED_COLOR, 2)
            cv2.line(result_frame, (x1, y1), (x1, y1 + corner_len), LOCKED_COLOR, 2)
            # Top-right
            cv2.line(result_frame, (x2, y1), (x2 - corner_len, y1), LOCKED_COLOR, 2)
            cv2.line(result_frame, (x2, y1), (x2, y1 + corner_len), LOCKED_COLOR, 2)
            # Bottom-left
            cv2.line(result_frame, (x1, y2), (x1 + corner_len, y2), LOCKED_COLOR, 2)
            cv2.line(result_frame, (x1, y2), (x1, y2 - corner_len), LOCKED_COLOR, 2)
            # Bottom-right
            cv2.line(result_frame, (x2, y2), (x2 - corner_len, y2), LOCKED_COLOR, 2)
            cv2.line(result_frame, (x2, y2), (x2, y2 - corner_len), LOCKED_COLOR, 2)
    
    return result_frame

# ============= WORKER THREAD =============
def detection_worker(input_queue, output_queue, detector, face_recognizer, tracker, target_tracker):
    """Worker thread for detection and tracking"""
    while True:
        try:
            frame = input_queue.get()
            if frame is None:  # Poison pill to stop thread
                break
                
            # Detect faces
            detected_faces = detector.detect(frame)
            
            # Process face recognition if enabled
            if ENABLE_FACE_RECOGNITION and face_recognizer:
                for face in detected_faces:
                    x1, y1, x2, y2 = face['bbox']
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size > 0:  # Make sure ROI is not empty
                        face_name, is_locked = face_recognizer.recognize_face(face_roi)
                        face['face_name'] = face_name
                        face['lock_status'] = is_locked
            
            # Update tracking
            tracked_faces = tracker.update(frame, detected_faces)
            
            # Find targeting points
            targets = target_tracker.find_targets(frame, tracked_faces)
            
            # Send results back
            output_queue.put((frame, targets))
            
        except Exception as e:
            print(f"Error in detection thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            input_queue.task_done()

# ============= MAIN FUNCTION =============
def main():
    # Initialize components
    detector = FaceDetector()
    
    # Initialize face recognition if enabled
    face_recognizer = FaceRecognizer() if ENABLE_FACE_RECOGNITION else None
    
    # Initialize tracking
    tracker = PersonTracker(max_disappeared=MAX_DISAPPEARED)
    target_tracker = TargetTracker(offset_y=VERTICAL_OFFSET, smoothing=SMOOTHING_FACTOR)
    
    # Open video source
    if USE_VIDEO_FILE and VIDEO_FILE_PATH:
        print(f"Using video file: {VIDEO_FILE_PATH}")
        video_source = VIDEO_FILE_PATH
    else:
        print(f"Using IP camera: {CAMERA_IP}")
        video_source = CAMERA_IP
        
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Create queues for threaded processing
    frame_queue = queue.Queue(maxsize=1)  # Only keep most recent frame
    result_queue = queue.Queue(maxsize=2)
    
    # Start worker thread
    worker = threading.Thread(target=detection_worker, 
                          args=(frame_queue, result_queue, detector, 
                               face_recognizer, tracker, target_tracker))
    worker.daemon = True
    worker.start()
    
    # Performance tracking
    prev_time = time.time()
    frame_count = 0
    last_targets = []
    
    print("Starting main loop. Press 'q' to quit, 'r' to reset face lock.")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            # If using video file, we've reached the end
            if USE_VIDEO_FILE:
                print("End of video file")
                break
            time.sleep(0.1)
            continue
        
        # Resize for better performance
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Process every Nth frame for better performance
        process_this_frame = frame_count % (SKIP_FRAMES + 1) == 0
        
        if process_this_frame:
            # Clear queue if full (discard old frames)
            while frame_queue.full():
                try:
                    frame_queue.get_nowait()
                    frame_queue.task_done()
                except:
                    break
                    
            # Add new frame to processing queue
            frame_queue.put(frame.copy())
        
        # Check if we have processed results
        try:
            if not result_queue.empty():
                _, last_targets = result_queue.get_nowait()
        except:
            pass
        
        # Create visual overlay with latest targets
        result = draw_targeting_overlay(frame, last_targets)
        
        # Calculate and display FPS
        if SHOW_FPS:
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            
            cv2.putText(result, f"FPS: {int(fps)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display result
        cv2.imshow("Targeting System", result)
        
        # Update frame counter
        frame_count += 1
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and face_recognizer:
            print("Resetting face lock")
            face_recognizer.clear_lock()
    
    # Clean up
    frame_queue.put(None)  # Signal thread to exit
    worker.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create known_faces directory if it doesn't exist
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    main()