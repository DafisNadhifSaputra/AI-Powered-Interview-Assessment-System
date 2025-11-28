"""
Video Downloader Module
Downloads videos from Google Drive URLs
"""

import os
import re
import tempfile
import gdown
from typing import Optional, Tuple


def extract_file_id(url: str) -> Optional[str]:
    """
    Extract Google Drive file ID from various URL formats.
    
    Supports:
    - https://drive.google.com/file/d/FILE_ID/view
    - https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    - https://drive.google.com/open?id=FILE_ID
    - https://drive.google.com/uc?id=FILE_ID
    """
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/d/([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_video(url: str, output_dir: Optional[str] = None) -> Tuple[str, bool]:
    """
    Download video from Google Drive URL.
    
    Args:
        url: Google Drive sharing URL
        output_dir: Optional directory for output (uses temp dir if None)
    
    Returns:
        Tuple of (file_path, success)
    """
    try:
        # Extract file ID
        file_id = extract_file_id(url)
        if not file_id:
            return "", False
        
        # Create output path
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        output_path = os.path.join(output_dir, f"{file_id}.mp4")
        
        # Build download URL
        download_url = f"https://drive.google.com/uc?id={file_id}"
        
        # Download using gdown with fuzzy URL parsing
        downloaded_path = gdown.download(
            url=download_url,
            output=output_path,
            quiet=False,
            fuzzy=True
        )
        
        if downloaded_path and os.path.exists(downloaded_path):
            return downloaded_path, True
        else:
            return "", False
            
    except Exception as e:
        print(f"Error downloading video: {e}")
        return "", False


def cleanup_video(file_path: str) -> bool:
    """
    Remove downloaded video file.
    
    Args:
        file_path: Path to video file
    
    Returns:
        True if cleanup successful
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            # Try to remove parent temp directory if empty
            parent_dir = os.path.dirname(file_path)
            if parent_dir and os.path.exists(parent_dir):
                try:
                    os.rmdir(parent_dir)
                except OSError:
                    pass  # Directory not empty, that's fine
            return True
        return False
    except Exception as e:
        print(f"Error cleaning up video: {e}")
        return False


if __name__ == "__main__":
    # Test with a sample URL
    test_url = "https://drive.google.com/file/d/1ABC123/view?usp=sharing"
    file_id = extract_file_id(test_url)
    print(f"Extracted file ID: {file_id}")
