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

### Project Structure

Ensure the following files and directories are present:

- `Dockerfile`: Defines the Streamlit app container
- `docker-compose.yml`: Configures services
- `requirements.txt`: Lists Python dependencies
- `app.py`: Main Streamlit application script
- `models/`: Directory containing model files
- `data/videos/`: Directory for input/output videos
- `data/logs/`: Directory for detection logs
- `feedback.json`: File for storing feedback

Example structure:

```
dispatch-monitoring-system/
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── models/
│   ├── yolov12.pt
│   ├── dish_classifier.pt
│   ├── tray_classifier.pt
├── data/
│   ├── videos/
│   ├── logs/
├── feedback.json
└── README.md
```

### Setup Instructions

1. **Create a `Dockerfile`**

   If not already present, create a `Dockerfile` in the project root:

   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Create a `docker-compose.yml`**

   Add a `docker-compose.yml` file in the project root:

   ```yaml
   version: "3.8"
   services:
     app:
       build:
         context: .
         dockerfile: Dockerfile
       ports:
         - "8501:8501"
       volumes:
         - ./data/videos:/app/data/videos
         - ./data/logs:/app/data/logs
         - ./feedback.json:/app/feedback.json
       environment:
         - PYTHONUNBUFFERED=1
       restart: unless-stopped
   volumes:
     videos:
     logs:
   ```

   **Optional**: To include a PostgreSQL database for feedback storage, use:

   ```yaml
   version: "3.8"
   services:
     app:
       build:
         context: .
         dockerfile: Dockerfile
       ports:
         - "8501:8501"
       volumes:
         - ./data/videos:/app/data/videos
         - ./data/logs:/app/data/logs
         - ./feedback.json:/app/feedback.json
       environment:
         - PYTHONUNBUFFERED=1
         - DB_HOST=db
         - DB_NAME=eatlab
         - DB_USER=admin
         - DB_PASSWORD=secret
       depends_on:
         - db
       restart: unless-stopped
     db:
       image: postgres:13
       environment:
         - POSTGRES_DB=eatlab
         - POSTGRES_USER=admin
         - POSTGRES_PASSWORD=secret
       volumes:
         - pgdata:/var/lib/postgresql/data
       ports:
         - "5432:5432"
       restart: unless-stopped
   volumes:
     videos:
     logs:
     pgdata:
   ```

3. **Prepare the Environment**

   - Ensure `requirements.txt` includes:
     ```
     streamlit==1.30.0
     ultralytics==8.3.0
     torch==2.0.1
     torchvision==0.15.2
     opencv-python==4.10.0
     pandas==2.2.2
     matplotlib==3.9.2
     numpy==1.26.4
     ```
   - Place model files in `models/`.
   - Create directories and feedback file:
     ```bash
     mkdir -p data/videos data/logs
     touch feedback.json
     echo "{}" > feedback.json
     ```

4. **Build and Run**

   In the project directory, run:

   ```bash
   docker-compose build
   docker-compose up -d
   ```

5. **Access the Application**

   Open `http://localhost:8501` in your browser to use the Streamlit UI.

6. **Stop the Application**

   ```bash
   docker-compose down
   ```

7. **Troubleshooting**

   - **Port Conflict**: Change `8501` in `docker-compose.yml` (e.g., `"8502:8501"`).
   - **Logs**: Check container logs:
     ```bash
     docker-compose logs app
     ```
   - **Permissions**: Set read/write permissions:
     ```bash
     chmod -R 777 data/ feedback.json
     ```

8. **Optional: Database Setup**

   If using PostgreSQL, create a feedback table:

   ```bash
   docker-compose exec db psql -U admin -d eatlab -c "
   CREATE TABLE feedback (
       id SERIAL PRIMARY KEY,
       video VARCHAR(255),
       frame INTEGER,
       track_id INTEGER,
       feedback VARCHAR(100),
       comment TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );"
   ```

   Update `app.py` to use the database with environment variables (`DB_HOST`, `DB_NAME`, etc.).

### Notes

- **Models**: Ensure model compatibility with `ultralytics` and `torch` versions.
- **GPU Support**: Use a CUDA-enabled base image (e.g., `nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04`) for GPU acceleration.
- **Storage**: Clean up `data/` directories periodically to manage disk space.
- **Security**: In production, secure database credentials and use a reverse proxy (e.g., Nginx) for the app.

## Example Usage

1. Start the application:
   ```bash
   docker-compose up -d
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
