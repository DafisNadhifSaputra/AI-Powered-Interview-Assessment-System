from faster_whisper import WhisperModel
from typing import Optional, Tuple, List
import os

# Global model instance for reuse
_model: Optional[WhisperModel] = None
_current_model_size: Optional[str] = None


def get_whisper_model(model_size: str = "large-v3-turbo") -> WhisperModel:
    global _model, _current_model_size
    
    if _model is None or _current_model_size != model_size:
        print(f"Loading Whisper model: {model_size}")
        try:
            _model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",  # Use int8 quantization for CPU efficiency
                download_root=None    # Use default cache directory
            )
            _current_model_size = model_size
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise
    
    return _model


def transcribe_video(
    video_path: str,
    language: Optional[str] = None,
    model_size: str = "large-v3-turbo"
) -> Tuple[str, dict]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    model = get_whisper_model(model_size)
    
    # Transcribe directly from video - config optimized for interview
    print(f"Transcribing: {video_path}")
    segments, info = model.transcribe(
        video_path,
        language=language,
        vad_filter=True,
        vad_parameters=dict(
            threshold=0.3,
            min_speech_duration_ms=250,  
            max_speech_duration_s=30,    
            min_silence_duration_ms=300, 
            speech_pad_ms=200            
        ),
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=True,  
        no_speech_threshold=0.4
    )
    
    # Collect all segments, filter hallucinations
    segment_list = []
    full_text_parts = []
    
    for segment in segments:
        text = segment.text.strip()
        
        if len(text) < 3:
            continue
        
        # Detect repetition 
        words = text.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Too many repeated words
                print(f"  [SKIPPED - repetition]: {text[:50]}...")
                continue
        
        segment_data = {
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": text
        }
        segment_list.append(segment_data)
        full_text_parts.append(text)
    
    # Combine full transcript
    full_transcript = " ".join(full_text_parts)
    print(f"Transcription complete: {len(full_transcript)} characters, {len(segment_list)} segments")
    
    # Prepare metadata
    metadata = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "num_segments": len(segment_list),
        "segments": segment_list
    }
    
    return full_transcript, metadata


def transcribe_with_timestamps(
    video_path: str,
    language: Optional[str] = None,
    model_size: str = "small"
) -> List[dict]:
    _, metadata = transcribe_video(video_path, language, model_size)
    return metadata["segments"]


if __name__ == "__main__":
    # Test transcription
    print("Transcription module loaded")
    print("Usage: transcribe_video('path/to/video.mp4')")
