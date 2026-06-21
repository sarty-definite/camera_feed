# 24/7 NVR Camera Feed & AI Metadata Pipeline

A production-ready, zero-gap Network Video Recorder (NVR) capture system designed for continuous 24/7 operations. It slices RTSP video feeds into clean segments, runs concurrent YOLOv8 classification (object tracking) on keyframes to conserve CPU, logs events with sub-second timestamps, and uploads footage/metadata to Google Drive with automated quota pruning.

---

## Architecture Overview

```
                   +------------------------+
                   |   RTSP Camera Stream   |
                   +-----------+------------+
                               |
                        (FFmpeg Slicer)
                               |
                               v
                     +---------+--------+
                     |  storage/raw/    |
                     |  (Local Chunks)  |
                     +---------+--------+
                               |
                       (Directory Watcher)
                               |
                               v
                       +-------+-------+
                       |   ml_queue    |
                       +-------+-------+
                               |
                       (ML Worker Thread)
                               |
                               v
                       +-------+-------+
                       | upload_queue  |
                       +-------+-------+
                               |
                     (Upload Worker Thread)
                               |
                               +-----------------------------+
                               |                             |
                               v                             v
                   +-----------+------------+   +------------+-----------+
                   | GDrive: Video Segment  |   |  GDrive: Event Metadata|
                   | (e.g. YYYY/MM/DD/HH)   |   |  (Linked JSON offsets) |
                   +------------------------+   +------------------------+
```

---

## Features

1. **24/7 Continuous Recording**: Subprocess FFmpeg capture engine copying stream codes losslessly (`-c copy`) without re-encoding, minimizing CPU load.
2. **Resilient Auto-Reconnection**: Reconnection logic handles RTSP network drops and backs off exponentially (up to 60s delay) to prevent CPU thrashing.
3. **Decoupled Concurrency**: Separate worker queues and threads isolate CPU-heavy machine learning from network I/O uploads.
4. **Efficient AI Grabbing**: Uses OpenCV's `cap.grab()` to bypass video decompression for skipped frames, accelerating inference.
5. **Configurable Tracking Targets**: Configure classification filters via `.env` (supports Person, Vehicles, Pets, and Luggage).
6. **Smart Storage Pruning**: Aggregates all Google Drive logs and enforces a strict storage quota ceiling (default 64GB) by deleting oldest files first.
7. **Production Logging**: Rotating log handlers keep logs from eating local storage, rotating at 10MB file sizes.
8. **Graceful Terminations**: Catches exit signals (`SIGINT`, `SIGTERM`) to cleanly wind down background threads, close file handles, and terminate subprocesses.

---

## Setup & Installation

### 1. Install System Requirements
Ensure [FFmpeg](https://ffmpeg.org/) is installed and added to your system's PATH.

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Google Drive API Access
1. Visit the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a project and enable the **Google Drive API**.
3. Configure your OAuth Consent Screen and download the client configuration. Save it in the project root folder as `credentials.json`.
4. Run the script once manually to complete OAuth authentication. The session is cached in `token.json` for subsequent silent operations.

### 4. Configuration (`.env`)
Create or edit your `.env` file in the root directory:

```env
# Stream Details
RTSP_URL=rtsp://your_camera_ip:554/live/ch00_0
CHUNK_DURATION_SECONDS=240  # 4-minute chunks

# ML Configuration
ENABLE_ML=True
YOLO_MODEL_VERSION=yolov8n.pt
ML_FRAME_SAMPLE_RATE=60 # Process 1 out of 60 frames

# COCO Dataset target index filters (comma separated)
# Person=0, Bicycle=1, Car=2, Motorcycle=3, Cat=15, Dog=16, Backpack=24, Suitcase=28
YOLO_TARGET_CLASSES=0,2,15,16

# Cloud Storage ID
GOOGLE_DRIVE_FOLDER_ID=your_gdrive_folder_id

# Logging Config
LOG_LEVEL=INFO
```

---

## Operations & Usage

### Starting the Pipeline
Simply run:
```bash
python main.py
```

All logs will output to the terminal and rotate locally inside [nvr.log](file:///c:/Users/sarth/OneDrive/Desktop/quick-scripts/camera_feed/storage/nvr.log).

---

## Seeking & Searching Activity Logs
Since videos are saved chronologically using `%d%m%Y%H%M%S` and metadata logs map event offsets to that start time, searching for an event is highly efficient:

1. Locate the metadata JSON files in Google Drive matching the date/time range.
2. Read the `timestamp_offset_sec` within the JSON.
3. Open the corresponding `.mp4` file and seek exactly to the offset (e.g. at 45 seconds) to view the activity.
