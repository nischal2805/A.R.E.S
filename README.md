# YOLO Targeting System

## Overview
The YOLO Targeting System is a real-time application designed to detect humans in video frames using the YOLO (You Only Look Once) object detection model. It tracks the position of detected individuals' eyes to calculate the center of their foreheads, providing a visual overlay for targeting purposes.

## Features
- Real-time human detection using YOLO.
- Eye tracking to determine the forehead center.
- Visual overlays on detected faces with targeting circles.
- Smooth tracking with minimal lag.
- Compatible with IP camera streaming.

## Project Structure
```
yolo-targeting-system
├── src
│   ├── main.py                # Entry point for the application
│   ├── camera
│   │   ├── __init__.py        # Camera module initializer
│   │   ├── ip_camera.py       # IP camera streaming implementation
│   │   └── camera_utils.py     # Utility functions for camera operations
│   ├── detection
│   │   ├── __init__.py        # Detection module initializer
│   │   ├── yolo_detector.py    # YOLO object detection implementation
│   │   └── eye_tracker.py      # Eye tracking logic
│   ├── visualization
│   │   ├── __init__.py        # Visualization module initializer
│   │   └── overlay.py          # Drawing overlays on video frames
│   ├── utils
│   │   ├── __init__.py        # Utils module initializer
│   │   └── performance.py      # Performance measurement functions
│   └── config
│       ├── __init__.py        # Config module initializer
│       └── settings.py         # Configuration settings
├── models
│   └── .gitkeep               # Keeps models directory in version control
├── requirements.txt            # Project dependencies
├── .gitignore                  # Files to ignore in version control
└── README.md                   # Project documentation
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd yolo-targeting-system
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Ensure your IP camera is set up and accessible.
2. Run the application:
   ```
   python src/main.py
   ```

3. The application will start processing video frames, detecting humans, and displaying the targeting overlays.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.