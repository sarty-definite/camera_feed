import os
import subprocess
import time
import threading
import signal
import sys
import logging
from logging.handlers import RotatingFileHandler
from queue import Queue
from datetime import datetime
from dotenv import load_dotenv

from ml_worker import analyze_video
from gdrive_uploader import upload_file_to_drive

# Load config
load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")
CHUNK_TIME = os.getenv("CHUNK_DURATION_SECONDS", "240")
ENABLE_ML = os.getenv("ENABLE_ML", "False").lower() == "true"
RAW_DIR = "storage/raw"
LOG_DIR = "storage"
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()

# Ensure directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs("storage/metadata", exist_ok=True)

# Set up Production logging with Rotation
log_level = getattr(logging, LOG_LEVEL_STR, logging.INFO)
logger = logging.getLogger("nvr")
logger.setLevel(log_level)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] (%(name)s) %(message)s')

# Rotating File Handler (10MB max per file, keep 5 backups)
file_handler = RotatingFileHandler(os.path.join(LOG_DIR, "nvr.log"), maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream Handler (console output)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("Initializing NVR Camera Feed System...")

# Queues
ml_queue = Queue()
upload_queue = Queue()

# Thread control event
shutdown_event = threading.Event()

def ml_worker():
    """Worker thread that processes video clips with YOLO analysis."""
    logger.info("ML analysis background thread initialized and listening...")
    while not shutdown_event.is_set():
        try:
            video_path = ml_queue.get(timeout=2.0)
            if video_path is None:
                ml_queue.task_done()
                break
            
            logger.info(f"ML worker received segment: {video_path}")
            metadata_path = None
            
            if ENABLE_ML:
                try:
                    metadata_path = analyze_video(video_path)
                except Exception as e:
                    logger.error(f"Uncaught error in ML analysis for {video_path}: {e}", exc_info=True)
            
            # Pass on to the upload queue even if ML failed/skipped
            upload_queue.put((video_path, metadata_path))
            ml_queue.task_done()
        except Exception:
            # timeout raises queue.Empty, which is ignored so we can loop & check shutdown_event
            pass
    logger.info("ML background worker thread shut down.")

def upload_worker():
    """Worker thread that uploads video segments and event logs to Google Drive."""
    logger.info("Google Drive uploader background thread initialized and listening...")
    while not shutdown_event.is_set():
        try:
            item = upload_queue.get(timeout=2.0)
            if item is None:
                upload_queue.task_done()
                break
            
            video_path, metadata_path = item
            logger.info(f"Uploader worker received segment: {video_path}")
            
            try:
                upload_file_to_drive(video_path, metadata_path)
            except Exception as e:
                logger.error(f"Uncaught error in GDrive upload for {video_path}: {e}", exc_info=True)
                
            upload_queue.task_done()
        except Exception:
            pass
    logger.info("Google Drive uploader background worker thread shut down.")

def watch_directory():
    """Monitors raw storage folder and queues segments as soon as FFmpeg rotates to a new segment."""
    logger.info("Directory watcher thread initialized...")
    seen_files = set()
    
    while not shutdown_event.is_set():
        try:
            # Sort files chronologically
            files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.mp4')])
            
            # We need at least 2 files to guarantee the previous one is completely written and closed
            if len(files) > 1:
                for file in files[:-1]:  
                    full_path = os.path.join(RAW_DIR, file)
                    
                    if full_path not in seen_files:
                        logger.info(f"Watcher detected fully finalized segment: {file}")
                        seen_files.add(full_path)
                        ml_queue.put(full_path)
                        logger.debug(f"Segment forwarded to ML queue: {file}")
            
            # Garbage collect deleted files from seen_files to prevent memory leaks
            current_paths = {os.path.join(RAW_DIR, f) for f in files}
            seen_files.intersection_update(current_paths)
            
        except Exception as e:
            logger.error(f"Directory watcher loop encountered an error: {e}", exc_info=True)
            
        time.sleep(5)
    logger.info("Directory watcher thread shut down.")

# Global references for cleanup
ffmpeg_proc = None
ffmpeg_log_file = None

def start_rtsp_slicer():
    """Spawns an FFmpeg subprocess to record and slice the RTSP stream."""
    global ffmpeg_log_file
    output_template = os.path.join(RAW_DIR, "%d%m%Y%H%M%S.mp4")
    log_path = os.path.join(LOG_DIR, "ffmpeg.log")

    cmd = [
        'ffmpeg', '-y',
        '-rtsp_transport', 'tcp',
        '-i', RTSP_URL,
        '-c', 'copy',
        '-map', '0',
        '-f', 'segment',
        '-segment_time', CHUNK_TIME,
        '-segment_format', 'mp4',
        '-reset_timestamps', '1',
        '-strftime', '1',
        output_template
    ]
    
    logger.info("Initiating zero-gap background capture engine (FFmpeg)...")
    if ffmpeg_log_file is not None:
        try:
            ffmpeg_log_file.close()
        except Exception as e:
            logger.warning(f"Failed to close legacy FFmpeg log handle: {e}")
            
    try:
        ffmpeg_log_file = open(log_path, "a")
        process = subprocess.Popen(cmd, stdout=ffmpeg_log_file, stderr=subprocess.STDOUT)
        logger.info(f"FFmpeg capture engine started successfully (PID: {process.pid})")
        return process
    except Exception as e:
        logger.critical(f"Failed to launch FFmpeg capture engine: {e}", exc_info=True)
        return None

def handle_shutdown(signum, frame):
    """Gracefully terminates background processes and threads on OS signals."""
    global ffmpeg_proc, ffmpeg_log_file
    logger.warning(f"Received termination signal ({signum}). Initiating graceful shutdown...")
    
    shutdown_event.set()
    
    # Terminate FFmpeg Capture Process
    if ffmpeg_proc and ffmpeg_proc.poll() is None:
        logger.info("Terminating FFmpeg capture process...")
        try:
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=10)
            logger.info("FFmpeg capture process cleanly terminated.")
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg did not terminate. Killing process...")
            ffmpeg_proc.kill()
        except Exception as e:
            logger.error(f"Error terminating FFmpeg: {e}")

    # Close FFmpeg Log File
    if ffmpeg_log_file:
        try:
            ffmpeg_log_file.close()
        except Exception:
            pass

    # Stop background workers by placing stop signals in the queues
    ml_queue.put(None)
    upload_queue.put(None)
    
    logger.info("System shutdown sequence finished. Exiting process.")
    sys.exit(0)

# Register signals
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

if __name__ == "__main__":
    ml_thread = threading.Thread(target=ml_worker, name="MLWorkerThread", daemon=True)
    upload_thread = threading.Thread(target=upload_worker, name="UploadWorkerThread", daemon=True)
    watcher_thread = threading.Thread(target=watch_directory, name="WatcherThread", daemon=True)
    
    ml_thread.start()
    upload_thread.start()
    watcher_thread.start()
    
    ffmpeg_proc = start_rtsp_slicer()
    
    consecutive_failures = 0
    try:
        while True:
            if ffmpeg_proc is None or ffmpeg_proc.poll() is not None:
                consecutive_failures += 1
                backoff_delay = min(5 * consecutive_failures, 60)
                logger.error(f"Capture stream dropped (Consecutive failures: {consecutive_failures}). Re-initializing in {backoff_delay}s...")
                time.sleep(backoff_delay)
                ffmpeg_proc = start_rtsp_slicer()
            else:
                consecutive_failures = 0
            time.sleep(5)
    except KeyboardInterrupt:
        # KeyboardInterrupt will trigger handle_shutdown due to signal registering, 
        # but keep this block as safety fallback
        handle_shutdown(signal.SIGINT, None)