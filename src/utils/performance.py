import time
def measure_performance(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Performance: {func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper

def optimize_frame_processing(frame):
    # Placeholder for optimization logic
    # This could include resizing the frame, adjusting color channels, etc.
    return frame

def calculate_fps(frame_count, start_time):
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
        return fps
    return 0