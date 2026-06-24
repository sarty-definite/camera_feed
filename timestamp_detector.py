import cv2
import easyocr
import re
import logging
from datetime import datetime
import os
import warnings

# Suppress PyTorch user warnings about pin_memory or other non-critical accelerator warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

logger = logging.getLogger("nvr.timestamp_detector")

# Lazy loaded reader
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        logger.info("Initializing EasyOCR reader (CPU mode)...")
        # EasyOCR defaults to checking GPU, we force gpu=False for predictable CPU usage.
        # verbose=False suppresses the initialization prints to stdout.
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader

def parse_ocr_text(ocr_results):
    if not ocr_results:
        return None
    # Join text components
    text = " ".join([res[1] for res in ocr_results])
    # Normalize noise: keep only digits, letters, spaces, colons, hyphens, slashes, dots
    text = re.sub(r'[^a-zA-Z0-9\s\-\/\:\.]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove spaces around punctuation to unify things like "00 : 07 : 38" -> "00:07:38"
    text = re.sub(r'\s*([:\-\/\.])\s*', r'\1', text)
    
    # Standard format match: YYYY-MM-DD HH:MM:SS
    match = re.search(r'(\d{4}[-\/]\d{2}[-\/]\d{2})[\s_]+(\d{2}:\d{2}:\d{2})', text)
    if match:
        date_str = match.group(1).replace('/', '-')
        time_str = match.group(2)
        try:
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
            
    # Fallback 1: replace dots/hyphens in time with colons and retry
    # e.g., if OCR got "2026-06-24 00.07.38"
    match_dots = re.search(r'(\d{4}[-\/]\d{2}[-\/]\d{2})[\s_]+(\d{2})[\.\-](\d{2})[\.\-](\d{2})', text)
    if match_dots:
        date_str = match_dots.group(1).replace('/', '-')
        time_str = f"{match_dots.group(2)}:{match_dots.group(3)}:{match_dots.group(4)}"
        try:
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    # Fallback 2: parse digits only
    digits = re.sub(r'\D', '', text)
    if len(digits) == 14:
        try:
            return datetime.strptime(digits, "%Y%m%d%H%M%S")
        except ValueError:
            pass

    if len(digits) >= 14:
        # Search for first 14-digit block starting with "20"
        for i in range(len(digits) - 13):
            slice_str = digits[i:i+14]
            if slice_str.startswith("20"):
                try:
                    return datetime.strptime(slice_str, "%Y%m%d%H%M%S")
                except ValueError:
                    pass

    logger.warning(f"Could not parse timestamp from OCR text: '{text}' (digits: '{digits}')")
    return None

def detect_timestamps(video_path):
    """
    Opens the video file, extracts the first and last frame, crops the top right corner
    where the overlay timestamp resides, performs OCR, and parses the timestamps.
    
    Returns a dict with:
        "first_frame_timestamp": str or None,
        "last_frame_timestamp": str or None,
        "duration_seconds": float or None
    """
    logger.info(f"Extracting overlay timestamps for: {os.path.basename(video_path)}")
    
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        logger.error(f"Video file is missing or empty: {video_path}")
        return {
            "first_frame_timestamp": None,
            "last_frame_timestamp": None,
            "duration_seconds": None
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file (corrupted): {video_path}")
        return {
            "first_frame_timestamp": None,
            "last_frame_timestamp": None,
            "duration_seconds": None
        }
        
    # Get total frames and FPS to estimate duration and locate frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0 or fps <= 0:
        logger.error(f"Video file appears to be corrupted (invalid frame count or FPS): {video_path}")
        cap.release()
        return {
            "first_frame_timestamp": None,
            "last_frame_timestamp": None,
            "duration_seconds": None
        }

    duration = round(total_frames / fps, 2)
    first_dt = None
    last_dt = None
    
    # 1. Read first frame (with fallback: 0, 3, 6, 9, ...)
    max_first_frame_search = min(total_frames, 90)
    for idx in range(0, max_first_frame_search, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            continue
            
        h, w, _ = first_frame.shape
        crop_h = int(h * 0.1)
        crop_w = int(w * 0.35)
        crop_first = first_frame[0:crop_h, (w - crop_w):w]
        
        try:
            import torch
            reader = get_reader()
            with torch.no_grad():
                results = reader.readtext(crop_first)
            dt = parse_ocr_text(results)
            if dt is not None:
                first_dt = dt
                break
        except Exception as e:
            logger.error(f"OCR failed on frame {idx} of {os.path.basename(video_path)}: {e}", exc_info=True)

    # 2. Read last frame (with fallback: pos, pos-3, pos-6, ...)
    start_pos = max(0, total_frames - 5)
    min_pos = max(0, start_pos - 90)
    for idx in range(start_pos, min_pos - 1, -3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, last_frame = cap.read()
        if not ret or last_frame is None:
            continue
            
        h, w, _ = last_frame.shape
        crop_h = int(h * 0.1)
        crop_w = int(w * 0.35)
        crop_last = last_frame[0:crop_h, (w - crop_w):w]
        
        try:
            import torch
            reader = get_reader()
            with torch.no_grad():
                results = reader.readtext(crop_last)
            dt = parse_ocr_text(results)
            if dt is not None:
                last_dt = dt
                break
        except Exception as e:
            logger.error(f"OCR failed on frame {idx} of {os.path.basename(video_path)}: {e}", exc_info=True)

    cap.release()
    
    # If we successfully parsed both timestamps, compute exact duration from them as backup/refinement
    if first_dt and last_dt:
        ocr_duration = (last_dt - first_dt).total_seconds()
        if ocr_duration > 0:
            duration = ocr_duration

    return {
        "first_frame_timestamp": first_dt.strftime("%Y-%m-%d %H:%M:%S") if first_dt else None,
        "last_frame_timestamp": last_dt.strftime("%Y-%m-%d %H:%M:%S") if last_dt else None,
        "duration_seconds": duration
    }
