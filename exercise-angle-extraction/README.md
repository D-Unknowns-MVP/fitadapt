# Exercise Angle Extraction

This project uses MediaPipe Pose Landmarker to detect body landmarks from exercise videos and calculate joint angles.

## Features
- Detects body landmarks
- Calculates shoulder, elbow, hip, knee, and ankle angles
- Calculates body-segment angles relative to the ground
- Exports extracted angles into a CSV file

## Requirements
- Python
- OpenCV
- MediaPipe
- NumPy

## Installation
Install the required libraries:

pip install opencv-python mediapipe numpy

## Usage
1. Place the MediaPipe model file in the project folder.
2. Place your exercise video in the videos folder.
3. Configure the video path and output CSV filename in the Python script.
4. Run the extraction script.

The output is a CSV file containing the extracted joint angles.
