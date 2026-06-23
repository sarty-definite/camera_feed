import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

load_dotenv()
logger = logging.getLogger("nvr.gdrive_uploader")

SCOPES = ['https://www.googleapis.com/auth/drive.file']
ROOT_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
MAX_STORAGE_BYTES = 64 * 1024 * 1024 * 1024  # Strict 64 GB limit

_service_cache = None

def get_gdrive_service():
    """Initializes and returns Google Drive API service using authorization tokens."""
    global _service_cache
    if _service_cache is not None:
        return _service_cache

    creds = None
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except Exception as e:
            logger.warning(f"Failed to load credentials from token.json: {e}. Re-authenticating...")

    # If credentials are not valid or not present, handle refresh or prompt auth
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logger.debug("Credentials expired. Attempting refresh token...")
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Failed to refresh credential token: {e}. Re-triggering authorization flow...")
                creds = None
        
        if not creds:
            if not os.path.exists('credentials.json'):
                logger.critical("credentials.json client secret file is missing! Google Drive uploads will fail.")
                raise FileNotFoundError("credentials.json missing")
            
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
        try:
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
            logger.debug("Saved fresh authorized session to token.json.")
        except Exception as e:
            logger.error(f"Failed to write fresh token.json file: {e}")

    _service_cache = build('drive', 'v3', credentials=creds)
    return _service_cache

def get_or_create_subfolder(service, parent_id, folder_name):
    """Finds or builds a folder layer to guarantee strict YYYY/MM/DD/HH structures."""
    query = f"name = '{folder_name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    
    # Retry up to 3 times on transient network failures
    for attempt in range(3):
        try:
            results = service.files().list(q=query, fields="files(id)").execute()
            files = results.get('files', [])
            if files:
                return files[0]['id']
            break
        except Exception as e:
            logger.warning(f"Error querying folder '{folder_name}' (Attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    else:
        logger.error(f"Failed folder query after 3 attempts: '{folder_name}'")
        raise IOError(f"Failed to query folder '{folder_name}' after 3 attempts due to network/API errors.")
    
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    
    for attempt in range(3):
        try:
            folder = service.files().create(body=folder_metadata, fields='id').execute()
            return folder.get('id')
        except Exception as e:
            logger.warning(f"Error creating folder '{folder_name}' (Attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
            
    logger.error(f"Critical: Failed to create folder '{folder_name}' after 3 attempts.")
    raise IOError(f"Could not resolve folder '{folder_name}' in Google Drive.")

def resolve_hourly_path(service):
    """Builds nested folder structure YYYY/MM/DD/HH and returns final hour folder ID."""
    now = datetime.now()
    year_id = get_or_create_subfolder(service, ROOT_FOLDER_ID, now.strftime("%Y"))
    month_id = get_or_create_subfolder(service, year_id, now.strftime("%m"))
    day_id = get_or_create_subfolder(service, month_id, now.strftime("%d"))
    hour_id = get_or_create_subfolder(service, day_id, now.strftime("%H"))
    return hour_id

def prune_storage_if_full(service):
    """Enforces strict 64GB boundary by dropping oldest objects and cleaning dead structures."""
    logger.debug("Reviewing Google Drive storage quota boundaries...")
    
    query = f"mimeType != 'application/vnd.google-apps.folder' and trashed = false"
    files = []
    page_token = None
    
    try:
        while True:
            results = service.files().list(
                q=query, 
                spaces='drive', 
                fields="nextPageToken, files(id, size, name, parents)", 
                pageSize=1000,
                pageToken=page_token
            ).execute()
            files.extend(results.get('files', []))
            page_token = results.get('nextPageToken')
            if not page_token:
                break
    except Exception as e:
        logger.error(f"Failed to scan cloud files list for quota enforcement: {e}", exc_info=True)
        return
        
    total_size = sum(int(f.get('size', 0)) for f in files)
    logger.debug(f"Current cloud storage utilization: {total_size / (1024**3):.2f} GB / {MAX_STORAGE_BYTES / (1024**3):.2f} GB limit.")
    
    if total_size < MAX_STORAGE_BYTES:
        return
        
    logger.warning(f"Storage limit crossed: {total_size / (1024**3):.2f} GB used. Pruning oldest archives...")
    
    # Sort files by name chronologically (standard camera naming system makes this easy)
    files.sort(key=lambda x: x['name'])
    
    while total_size >= MAX_STORAGE_BYTES and files:
        oldest_file = files.pop(0)
        file_id = oldest_file['id']
        file_size = int(oldest_file.get('size', 0))
        parent_id = oldest_file.get('parents', [None])[0]
        
        try:
            service.files().delete(fileId=file_id).execute()
            total_size -= file_size
            logger.debug(f"Deleted legacy cloud node: {oldest_file['name']} ({file_size / (1024**2):.2f} MB)")
            
            # Wipe out empty parent folders if all child files are gone
            if parent_id:
                siblings = service.files().list(q=f"'{parent_id}' in parents and trashed = false", fields="files(id)").execute()
                if not siblings.get('files', []):
                    service.files().delete(fileId=parent_id).execute()
                    logger.debug("Removed empty hourly directory wrapper.")
        except Exception as e:
            logger.error(f"Error purging file {file_id} ({oldest_file['name']}): {e}", exc_info=True)

def upload_file_to_drive(video_path, metadata_path=None):
    """Uploads video and metadata payload to resolved folder structure in Google Drive."""
    try:
        service = get_gdrive_service()
        
        # Enforce quota ceiling before adding new loads
        prune_storage_if_full(service)
        
        # Resolve target nested folder layer (YYYY/MM/DD/HH)
        target_folder_id = resolve_hourly_path(service)
        
        # Upload video payload
        file_metadata = {'name': os.path.basename(video_path), 'parents': [target_folder_id]}
        media = MediaFileUpload(video_path, mimetype='video/mp4', resumable=True)
        
        uploaded_video = None
        # Resumable upload attempt loop
        for attempt in range(3):
            try:
                uploaded_video = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                break
            except Exception as e:
                logger.warning(f"Failed video upload (Attempt {attempt+1}/3): {e}")
                time.sleep(5 * (attempt + 1))
        
        del media
        if not uploaded_video:
            raise IOError(f"Could not upload video segment: {video_path}")
        
        # Upload linked metadata payload
        uploaded_meta_success = False
        if metadata_path and os.path.exists(metadata_path):
            meta_metadata = {'name': os.path.basename(metadata_path), 'parents': [target_folder_id]}
            
            for attempt in range(3):
                try:
                    media_meta = MediaFileUpload(metadata_path, mimetype='application/json')
                    service.files().create(body=meta_metadata, media_body=media_meta).execute()
                    uploaded_meta_success = True
                    del media_meta
                    break
                except Exception as e:
                    logger.warning(f"Failed metadata upload (Attempt {attempt+1}/3): {e}")
                    time.sleep(2 * (attempt + 1))

        # Log a single info line with filename, time, and metadata filename if applicable
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vname = os.path.basename(video_path)
        if uploaded_meta_success and metadata_path:
            mname = os.path.basename(metadata_path)
            logger.info(f"Uploaded {vname} (with metadata {mname}) to Google Drive at {upload_time}")
        else:
            logger.info(f"Uploaded {vname} to Google Drive at {upload_time}")

    except Exception as e:
        logger.error(f"Upload exception encountered: {e}", exc_info=True)
        global _service_cache
        _service_cache = None
        return
        
    # Local cleanups
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.debug(f"Removed local video chunk: {video_path}")
        if metadata_path and uploaded_meta_success and os.path.exists(metadata_path):
            os.remove(metadata_path)
            logger.debug(f"Removed local metadata file: {metadata_path}")
    except Exception as e:
        logger.warning(f"Windows file lock pending for {os.path.basename(video_path)}. Postponing local deletion... ({e})")