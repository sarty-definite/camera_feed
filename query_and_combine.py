import os
import subprocess
import shutil
import io
from dotenv import load_dotenv
from googleapiclient.http import MediaIoBaseDownload
from gdrive_uploader import get_gdrive_service

load_dotenv()

# Load Google Drive Query from env. 
# Example query: name contains '.mp4' and name contains '21062026' and trashed = false
GDRIVE_QUERY = os.getenv("GDRIVE_QUERY", "mimeType = 'video/mp4' and trashed = false")
OUTPUT_FILENAME = os.getenv("COMBINED_OUTPUT_PATH", "storage/combined_output.mp4")
TEMP_DIR = "storage/temp_download"

def download_file(service, file_id, destination_path):
    """Downloads a file from Google Drive to local storage."""
    print(f"[*] Downloading file ID {file_id} to {destination_path}...")
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(destination_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"    Download progress: {int(status.progress() * 100)}%", end="\r")
        print("    Download complete.            ")

def combine_videos(video_paths, output_path):
    """Combines list of local video files losslessly using FFmpeg concat demuxer."""
    if not video_paths:
        print("[!] No video files to combine.")
        return False

    print(f"[*] Combining {len(video_paths)} videos into {output_path}...")
    
    # Create the text file listing all videos for FFmpeg's concat demuxer
    concat_list_path = os.path.join(TEMP_DIR, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for path in video_paths:
            # Absolute path needed or forward slashes for FFmpeg's safe parsing
            abs_path = os.path.abspath(path).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")

    # Run ffmpeg concat copy (lossless, no re-encoding, extremely fast)
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list_path,
        '-c', 'copy',
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"[✓] Successfully combined videos! Saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] FFmpeg combination failed: {e.stderr.decode('utf-8', errors='ignore')}")
        return False

def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    try:
        service = get_gdrive_service()
    except Exception as e:
        print(f"[!] Failed to authenticate with Google Drive: {e}")
        return

    print(f"[*] Searching Google Drive with query: \"{GDRIVE_QUERY}\"")
    
    try:
        # Fetch matching files
        results = service.files().list(
            q=GDRIVE_QUERY,
            fields="files(id, name, createdTime)",
            pageSize=100
        ).execute()
        files = results.get('files', [])
    except Exception as e:
        print(f"[!] Error fetching files from Google Drive: {e}")
        return

    if not files:
        print("[!] No matching files found in Google Drive.")
        return

    # Sort files chronologically by name (standard camera naming system makes this easy)
    files.sort(key=lambda x: x['name'])
    
    print(f"[+] Found {len(files)} matching files:")
    for idx, f in enumerate(files):
        print(f"  {idx + 1}. {f['name']} (ID: {f['id']})")

    local_paths = []
    try:
        # Download each file
        for f in files:
            local_path = os.path.join(TEMP_DIR, f['name'])
            download_file(service, f['id'], local_path)
            local_paths.append(local_path)

        # Merge the downloaded chunks
        combine_videos(local_paths, OUTPUT_FILENAME)
        
    finally:
        # Clean up temporary downloads
        print("[*] Cleaning up temporary files...")
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        print("[✓] Cleanup complete.")

if __name__ == "__main__":
    main()
