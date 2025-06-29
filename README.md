# Dispatch Monitoring System – EATLAB

This is an interactive Streamlit-based application for object tracking, fine-grained classification, and feedback-driven model improvement. The system leverages YOLOv12 for object detection and MobileNetV2 for classification of detected objects into detailed subcategories.

## Features Overview

### 1. Detection & Tracking (YOLOv12 + Streamlit UI)

- Upload `.mp4` videos and track objects (`tray`, `dish`) using a trained YOLOv12 model.
- Adjustable parameters via UI:
  - Confidence Threshold
  - IoU Threshold
  - Max Detections per Frame
  - Class Filter (by class ID)
- Output:
  - Annotated video with bounding boxes and sub-labels
  - Frame-level detection results as downloadable CSV
  - Auto-saved result videos and logs

### 2. Fine-Grained Classification

- After detection, each object is cropped and classified into:
  - `empty`, `kakigori`, or `not_empty`
- Separate classifiers (`dish_classifier.pt`, `tray_classifier.pt`) are used for each class.
- Classification results are shown inside bounding box labels:
  - Example: `dish (kakigori) ID:2 0.91`

### 3. Recently Tracked Videos

- List and preview of previously tracked result videos.
- Download completed videos directly from the dashboard.

### 4. Analytics Dashboard

- Visualize and analyze detection logs by class and sub-label.
- Key analytics:
  - Tray `empty` count per frame
  - Line chart showing frequency trends
- Download CSV reports of filtered data.

### 5. Feedback Collection System

- Store structured feedback from users in `feedback.json`:
  ```json
  {
    "video": "tracked_*.mp4",
    "frame": 123,
    "track_id": 5,
    "feedback": "Wrong ID",
    "comment": "Non Dish"
  }
  ```

## Running the Project with Docker Compose

This section explains how to set up and run the Dispatch Monitoring System using Docker Compose, which orchestrates the Streamlit application and optional services like a database for feedback or analytics.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (version 20.10 or higher)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 2.0 or higher)
- A `.mp4` video file for testing object detection and tracking
- Model files: `yolov12.pt`, `dish_classifier.pt`, `tray_classifier.pt`

## Example Usage

1. Start the application:
   ```bash
   docker-compose up
   ```
2. Open `http://localhost:8501` in your browser.
3. Upload a `.mp4` video to the Streamlit UI.
4. Adjust detection parameters (e.g., Confidence Threshold = 0.5, IoU Threshold = 0.4).
5. Download the annotated video and CSV results.
6. View analytics or submit feedback via the dashboard.
7. Stop the application:
   ```bash
   docker-compose down
   ```

## Contributing

Contributions are welcome! Please submit issues or pull requests to the project repository.

## License

This project is licensed under the MIT License.
