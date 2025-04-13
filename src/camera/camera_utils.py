import cv2
def capture_frame(camera):
    ret, frame = camera.read()
    if not ret:
        raise Exception("Failed to capture frame from camera.")
    return frame

def preprocess_frame(frame):
    # Resize frame to the required input size for YOLO
    resized_frame = cv2.resize(frame, (416, 416))
    # Normalize the pixel values
    normalized_frame = resized_frame / 255.0
    return normalized_frame

def release_camera(camera):
    camera.release()