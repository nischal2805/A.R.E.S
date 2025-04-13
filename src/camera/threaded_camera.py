import cv2
import threading
import time
import queue

class ThreadedCamera:
    def __init__(self, camera_url, width=320, height=240, queue_size=1):
        self.camera_url = camera_url
        self.width = width
        self.height = height
        self.stopped = False
        self.frame_queue = queue.Queue(maxsize=queue_size)
        
        # Start capture thread
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        
    def start(self):
        """Start the camera thread"""
        self.cap = cv2.VideoCapture(self.camera_url)
        if not self.cap.isOpened():
            print(f"Failed to open camera at {self.camera_url}")
            return False
            
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Start the thread
        self.thread.start()
        return True
        
    def _update(self):
        """Thread function to continuously grab frames"""
        while not self.stopped:
            if not self.cap.isOpened():
                print("Reconnecting to camera...")
                self.cap = cv2.VideoCapture(self.camera_url)
                time.sleep(1)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to get frame, retrying...")
                time.sleep(0.1)
                continue
                
            # Resize for performance
            frame = cv2.resize(frame, (self.width, self.height))
            
            # If queue is full, remove oldest frame
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                    
            # Add new frame
            self.frame_queue.put(frame)
            
    def get_frame(self):
        """Get the most recent frame"""
        if self.frame_queue.empty():
            return None
        return self.frame_queue.get()
        
    def release(self):
        """Stop the thread and release resources"""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()