import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import shutil
import time
from datetime import datetime, timedelta
from video_creation.utils.config_loader import VideoConfig

def fetch_recent_images(minutes=180):
    config = VideoConfig()
    downloads_dir = r"C:\Users\mgabbay\Downloads"
    now = time.time()
    cutoff = now - (minutes * 60)
    # Get all image files downloaded in the last `minutes`
    recent_images = [
        os.path.join(downloads_dir, f)
        for f in os.listdir(downloads_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp",".mp3"))
        and os.path.isfile(os.path.join(downloads_dir, f))
        and os.path.getmtime(os.path.join(downloads_dir, f)) >= cutoff
    ]
    # Create a new folder under content_workitems with current date and time
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = os.path.join(config.content_workitems, f"images_{dt_str}")
    os.makedirs(dest_dir, exist_ok=True)
    # Move/copy images to the new folder
    for img_path in recent_images:
        shutil.copy2(img_path, dest_dir)
    return dest_dir
