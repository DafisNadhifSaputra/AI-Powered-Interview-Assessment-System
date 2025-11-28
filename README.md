---
title: AI Interview Assessment System
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
license: mit
python_version: "3.10"
---

# AI-Powered Interview Assessment System

This system automatically analyzes interview videos using AI to provide objective scoring and feedback.

## Features

- **Speech Transcription**: Uses Faster Whisper to convert speech to text
- **Eye Tracking Analysis**: Uses MediaPipe to analyze eye contact and attention
- **AI Scoring**: Uses Google Gemini to evaluate responses and provide scores (0-4 scale)
- **Automated Decision**: Generates PASSED/REVIEW/FAILED recommendations

## How It Works

1. **Video Download**: Downloads interview videos from Google Drive URLs
2. **Transcription**: Extracts and transcribes audio using Faster Whisper (small model, int8)
3. **Eye Analysis**: Analyzes eye movement patterns using MediaPipe Face Landmarker
4. **AI Assessment**: Sends transcript + eye metrics to Gemini for scoring

## Input Format

```json
{
  "reviewChecklists": {
    "interviews": [
      {
        "positionId": 1,
        "question": "Your interview question here",
        "isVideoExist": true,
        "recordedVideoUrl": "https://drive.google.com/file/d/VIDEO_ID/view"
      }
    ]
  }
}
```

## Output Format

```json
{
  "success": true,
  "data": {
    "pastReviews": [
      {
        "assessorProfile": {
          "id": 1,
          "name": "AI Assessor"
        },
        "decision": "PASSED",
        "reviewedAt": "2025-11-27 10:00:00",
        "scoresOverview": {
          "project": 0,
          "interview": 80,
          "total": 80
        },
        "reviewChecklistResult": {
          "interviews": {
            "minScore": 0,
            "maxScore": 4,
            "scores": [
              {"id": 1, "score": 3}
            ]
          }
        },
        "notes": "Assessment summary"
      }
    ]
  }
}
```

## API Usage

```python
import requests

response = requests.post(
    "https://YOUR_SPACE/api/assess",
    json={"data": [your_json_string]}
)
result = response.json()
```

## Environment Variables

Set the following in Hugging Face Spaces secrets:

- `GEMINI_API_KEY`: Your Google Gemini API key

## Limitations (Free Tier)

- Video duration: 5-10 minutes recommended
- Processing time: 1-3 minutes per video
- Sequential processing (one video at a time)

## Tech Stack

- Gradio 4.44.0
- Faster Whisper 1.0.3 (Dafisns/whisper-turbo-multilingual-fleurs, int8 quantization)
- MediaPipe 0.10.14
- Google Gemini API
- OpenCV (headless)
