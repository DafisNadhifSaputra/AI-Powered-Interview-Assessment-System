"""
Modules package for AI Interview Assessment System
"""

from .video_downloader import download_video, cleanup_video, extract_file_id
from .transcription import transcribe_video, get_whisper_model
from .eye_tracking import analyze_video as analyze_eye_tracking
from .gemini_scorer import assess_interview, batch_assess_interviews

# Check if MediaPipe is available
try:
    from .eye_tracking import get_face_detector, MEDIAPIPE_AVAILABLE
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    get_face_detector = None

__all__ = [
    "download_video",
    "cleanup_video", 
    "extract_file_id",
    "transcribe_video",
    "get_whisper_model",
    "analyze_eye_tracking",
    "get_face_detector",
    "assess_interview",
    "batch_assess_interviews",
    "MEDIAPIPE_AVAILABLE"
]
