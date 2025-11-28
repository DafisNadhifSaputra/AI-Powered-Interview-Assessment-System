"""
Eye Tracking Module
Analyzes eye movement and gaze direction using MediaPipe Face Landmarker
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import os
import urllib.request

# Try to import MediaPipe (may not be available on all Python versions)
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Eye tracking will use fallback mode.")

# MediaPipe Face Mesh landmark indices for eyes
# Left eye landmarks
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Right eye landmarks  
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Iris landmarks (center points)
LEFT_IRIS_CENTER = 473   # Center of left iris
RIGHT_IRIS_CENTER = 468  # Center of right iris

# Eye corner landmarks for gaze calculation
LEFT_EYE_LEFT_CORNER = 263
LEFT_EYE_RIGHT_CORNER = 362
RIGHT_EYE_LEFT_CORNER = 33
RIGHT_EYE_RIGHT_CORNER = 133

# Global detector instance
_face_detector = None


def download_model_if_needed() -> str:
    """
    Download MediaPipe face landmarker model if not present.
    
    Returns:
        Path to model file
    """
    if not MEDIAPIPE_AVAILABLE:
        return ""
        
    model_path = "face_landmarker_v2_with_blendshapes.task"
    
    if not os.path.exists(model_path):
        print("Downloading MediaPipe Face Landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully")
    
    return model_path


def get_face_detector():
    """
    Get or initialize Face Landmarker.
    Uses lazy loading to save memory.
    
    Returns:
        FaceLandmarker instance or None if MediaPipe not available
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
        
    global _face_detector
    
    if _face_detector is None:
        model_path = download_model_if_needed()
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        _face_detector = vision.FaceLandmarker.create_from_options(options)
        print("Face Landmarker initialized")
    
    return _face_detector


def calculate_eye_aspect_ratio(landmarks: List, eye_indices: List[int]) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) for blink detection.
    
    Args:
        landmarks: Face landmarks
        eye_indices: Indices of eye landmarks
    
    Returns:
        EAR value (lower = more closed)
    """
    # Get vertical eye landmarks
    top = landmarks[eye_indices[1]]
    bottom = landmarks[eye_indices[5]]
    
    # Get horizontal eye landmarks
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[3]]
    
    # Calculate distances
    vertical = np.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
    horizontal = np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2)
    
    if horizontal == 0:
        return 0
    
    return vertical / horizontal


def calculate_gaze_ratio(landmarks: List, iris_center_idx: int, 
                         left_corner_idx: int, right_corner_idx: int) -> float:
    """
    Calculate horizontal gaze ratio.
    0 = looking left, 0.5 = center, 1 = looking right
    
    Args:
        landmarks: Face landmarks
        iris_center_idx: Index of iris center
        left_corner_idx: Index of left eye corner
        right_corner_idx: Index of right eye corner
    
    Returns:
        Gaze ratio (0-1)
    """
    iris = landmarks[iris_center_idx]
    left_corner = landmarks[left_corner_idx]
    right_corner = landmarks[right_corner_idx]
    
    # Calculate horizontal position of iris relative to eye width
    eye_width = right_corner.x - left_corner.x
    if eye_width == 0:
        return 0.5
    
    iris_position = (iris.x - left_corner.x) / eye_width
    return max(0, min(1, iris_position))


def analyze_frame(frame: np.ndarray, detector) -> Optional[Dict]:
    """
    Analyze a single frame for eye metrics.
    
    Args:
        frame: BGR image frame
        detector: FaceLandmarker instance
    
    Returns:
        Dictionary with eye metrics or None if no face detected
    """
    if not MEDIAPIPE_AVAILABLE or detector is None:
        return None
        
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect face landmarks
    result = detector.detect(mp_image)
    
    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return None
    
    landmarks = result.face_landmarks[0]
    
    # Calculate left eye gaze
    left_gaze = calculate_gaze_ratio(
        landmarks, LEFT_IRIS_CENTER,
        LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER
    )
    
    # Calculate right eye gaze
    right_gaze = calculate_gaze_ratio(
        landmarks, RIGHT_IRIS_CENTER,
        RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER
    )
    
    # Average gaze (0.5 = looking at camera)
    avg_gaze = (left_gaze + right_gaze) / 2
    
    # Calculate eye contact score (how close to center)
    # 1.0 = perfect eye contact, 0.0 = looking away
    eye_contact_score = 1.0 - abs(avg_gaze - 0.5) * 2
    
    # Get blendshapes if available
    blendshapes = {}
    if result.face_blendshapes and len(result.face_blendshapes) > 0:
        for bs in result.face_blendshapes[0]:
            blendshapes[bs.category_name] = bs.score
    
    return {
        "left_gaze": left_gaze,
        "right_gaze": right_gaze,
        "avg_gaze": avg_gaze,
        "eye_contact_score": eye_contact_score,
        "blendshapes": blendshapes
    }


def analyze_video(
    video_path: str,
    sample_rate: int = 3,   # Analyze every 3rd frame (~10 FPS for 30fps video)
    max_frames: int = 1000  # Maximum frames to analyze (covers ~5 min video)
) -> Dict:
    """
    Analyze eye movement throughout a video.
    
    Args:
        video_path: Path to video file
        sample_rate: Analyze every Nth frame (higher = faster but less accurate)
        max_frames: Maximum number of frames to analyze
    
    Returns:
        Dictionary with aggregated eye metrics
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Check if MediaPipe is available
    if not MEDIAPIPE_AVAILABLE:
        # Return fallback metrics
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        return {
            "face_detection_rate": 50.0,  # Fallback value
            "eye_contact_percentage": 70.0,  # Fallback value
            "gaze_stability": 75.0,  # Fallback value
            "attention_score": 70.0,  # Fallback value
            "looking_away_percentage": 30.0,  # Fallback value
            "video_duration": round(duration, 2),
            "frames_analyzed": 0,
            "analysis_notes": "MediaPipe not available - using fallback metrics"
        }
    
    detector = get_face_detector()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Collect metrics
    gaze_values = []
    eye_contact_scores = []
    frames_with_face = 0
    frames_analyzed = 0
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames
        if frame_idx % sample_rate == 0:
            frames_analyzed += 1
            
            metrics = analyze_frame(frame, detector)
            if metrics:
                frames_with_face += 1
                gaze_values.append(metrics["avg_gaze"])
                eye_contact_scores.append(metrics["eye_contact_score"])
            
            if frames_analyzed >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    
    # Calculate aggregate metrics
    if len(gaze_values) == 0:
        return {
            "face_detection_rate": 0,
            "eye_contact_percentage": 0,
            "gaze_stability": 0,
            "attention_score": 0,
            "looking_away_percentage": 0,
            "video_duration": duration,
            "frames_analyzed": frames_analyzed,
            "analysis_notes": "No face detected in video"
        }
    
    # Eye contact: percentage of time looking at camera (gaze near 0.5)
    eye_contact_threshold = 0.3  # Consider looking at camera if within this range
    eye_contact_frames = sum(1 for g in gaze_values if abs(g - 0.5) < eye_contact_threshold)
    eye_contact_percentage = (eye_contact_frames / len(gaze_values)) * 100
    
    # Gaze stability: inverse of standard deviation (less movement = more stable)
    gaze_std = np.std(gaze_values)
    gaze_stability = max(0, 1 - gaze_std * 2) * 100  # Convert to percentage
    
    # Average eye contact score
    avg_eye_contact = np.mean(eye_contact_scores) * 100
    
    # Looking away percentage
    looking_away_frames = sum(1 for g in gaze_values if abs(g - 0.5) >= eye_contact_threshold)
    looking_away_percentage = (looking_away_frames / len(gaze_values)) * 100
    
    # Overall attention score (weighted average)
    attention_score = (
        avg_eye_contact * 0.4 +
        gaze_stability * 0.3 +
        (100 - looking_away_percentage) * 0.3
    )
    
    return {
        "face_detection_rate": (frames_with_face / frames_analyzed) * 100,
        "eye_contact_percentage": round(eye_contact_percentage, 2),
        "gaze_stability": round(gaze_stability, 2),
        "attention_score": round(attention_score, 2),
        "looking_away_percentage": round(looking_away_percentage, 2),
        "avg_gaze_position": round(np.mean(gaze_values), 3),
        "gaze_variance": round(gaze_std, 3),
        "video_duration": round(duration, 2),
        "frames_analyzed": frames_analyzed,
        "frames_with_face": frames_with_face,
        "analysis_notes": "Analysis completed successfully"
    }


if __name__ == "__main__":
    print("Eye Tracking module loaded")
    print("Usage: analyze_video('path/to/video.mp4')")
