import cv2
import numpy as np
import time

class IPCamera:
    def __init__(self, camera_url, width=640, height=480):
        """
        Initialize IP camera connection
        
        Args:
            camera_url: URL to the IP camera stream
            width: Desired frame width
            height: Desired frame height
        """
        self.camera_url = camera_url
        self.width = width
        self.height = height
        self.cap = None
        self.connect()
    
    def connect(self):
        """Establish connection to the IP camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_url)
            if not self.cap.isOpened():
                print(f"Failed to connect to camera at {self.camera_url}")
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return True
        except Exception as e:
            print(f"Error connecting to camera: {e}")
            return False
    
    def get_frame(self):
        """Capture a frame from the IP camera"""
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return None
        
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame, attempting to reconnect...")
            if self.connect():
                ret, frame = self.cap.read()
                if not ret:
                    return None
            else:
                return None
        
        return cv2.resize(frame, (self.width, self.height))
    
    def release(self):
        """Release the camera resource"""
        if self.cap is not None:
            self.cap.release()