import os
import json
import logging
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("nvr.ml_worker")

MODEL_VER = os.getenv("YOLO_MODEL_VERSION", "yolov8n.pt")
SAMPLE_RATE = int(os.getenv("ML_FRAME_SAMPLE_RATE", "60"))

# Initializes YOLOv8 Model. Automatically downloads the model pt if not present locally
try:
    model = YOLO(MODEL_VER)
except Exception as e:
    logger.critical(f"Failed to load YOLO model: {e}", exc_info=True)
    raise

# COCO Dataset index map targets we want to catch (Person=0)
# Custom classes can be supplied via YOLO_TARGET_CLASSES env variable
target_classes_str = os.getenv("YOLO_TARGET_CLASSES", "0")
try:
    TARGET_CLASSES = [int(x.strip()) for x in target_classes_str.split(",") if x.strip()]
    logger.info(f"YOLO worker targets set to class indices: {TARGET_CLASSES}")
except ValueError:
    logger.warning("Failed to parse YOLO_TARGET_CLASSES from environment. Defaulting to Person [0].")
    TARGET_CLASSES = [0]

def analyze_video(video_path):
    """Processes a video block frame-by-frame using efficient keyframe grabbing and YOLO analysis."""
    logger.info(f"Starting ML analysis on video segment: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file for ML analysis: {video_path}")
        return None

    base_name = os.path.basename(video_path).replace(".mp4", "")
    metadata_path = f"storage/metadata/{base_name}_events.json"
    
    frame_idx = 0
    events_found = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    try:
        while cap.isOpened():
            if frame_idx % SAMPLE_RATE == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                results = model(frame, verbose=False)[0]
                
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    if class_id in TARGET_CLASSES:
                        timestamp_sec = round(frame_idx / fps, 2)
                        event = {
                            "class": results.names[class_id],
                            "confidence": float(box.conf[0]),
                            "timestamp_offset_sec": timestamp_sec
                        }
                        events_found.append(event)
                        logger.debug(f"Detected target event: {event}")
                        break  # One event per keyframe slice is enough to log context
            else:
                ret = cap.grab()
                if not ret:
                    break
                    
            frame_idx += 1
    except Exception as e:
        logger.error(f"Error occurred during frame analysis loop of {base_name}: {e}", exc_info=True)
    finally:
        cap.release()
    
    if events_found:
        try:
            with open(metadata_path, 'w') as f:
                json.dump({"video_source": base_name, "events": events_found}, f, indent=4)
            logger.info(f"AI found {len(events_found)} critical triggers inside {base_name}! Metadata saved.")
            return metadata_path
        except Exception as e:
            logger.error(f"Failed to save metadata JSON for {base_name}: {e}", exc_info=True)
            
    logger.info(f"ML analysis complete for {base_name} - no critical triggers detected.")
    return None