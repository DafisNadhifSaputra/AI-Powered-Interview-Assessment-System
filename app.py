"""
AI-Powered Interview Assessment System
Analyzes interview videos using Whisper transcription, MediaPipe eye tracking, and Gemini LLM scoring.
"""

import gradio as gr
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import traceback

# Import modules
from modules.video_downloader import download_video, cleanup_video
from modules.transcription import transcribe_video, get_whisper_model
from modules.eye_tracking import analyze_video as analyze_eye_tracking
from modules.gemini_scorer import assess_interview

# Pre-load Whisper model at startup to avoid timeout during processing
print("Pre-loading Whisper model...")
try:
    get_whisper_model()
    print("Whisper model ready!")
except Exception as e:
    print(f"Warning: Could not pre-load Whisper model: {e}")


# ============================================
# Core Processing Functions
# ============================================

def process_single_interview(
    position_id: int,
    question: str,
    video_url: str,
    progress_callback=None
) -> Dict:
    """
    Process a single interview video.
    
    Args:
        position_id: Position/question ID
        question: Interview question
        video_url: Google Drive video URL
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with processing results
    """
    video_path = None
    
    try:
        # Step 1: Download video
        if progress_callback:
            progress_callback(f"[{position_id}] Downloading video...")
        
        video_path, success = download_video(video_url)
        if not success:
            return {
                "id": position_id,
                "score": 0,
                "error": "Failed to download video from Google Drive",
                "transcript": "",
                "eye_metrics": {}
            }
        
        # Step 2: Transcribe audio
        if progress_callback:
            progress_callback(f"[{position_id}] Transcribing audio...")
        
        try:
            transcript, metadata = transcribe_video(video_path)
        except Exception as e:
            transcript = ""
            metadata = {"error": str(e)}
        
        # Step 3: Analyze eye movement
        if progress_callback:
            progress_callback(f"[{position_id}] Analyzing eye movement...")
        
        try:
            eye_metrics = analyze_eye_tracking(video_path, sample_rate=15, max_frames=300)
        except Exception as e:
            eye_metrics = {"error": str(e), "eye_contact_percentage": 0}
        
        # Step 4: Get Gemini assessment
        if progress_callback:
            progress_callback(f"[{position_id}] Getting AI assessment...")
        
        try:
            score, reasoning, full_assessment = assess_interview(
                question=question,
                transcript=transcript,
                eye_metrics=eye_metrics,
                position_id=position_id
            )
        except Exception as e:
            score = 2
            reasoning = f"Assessment error: {str(e)}"
            full_assessment = {"error": str(e)}
        
        return {
            "id": position_id,
            "score": score,
            "reasoning": reasoning,
            "transcript": transcript,
            "transcript_metadata": metadata,
            "eye_metrics": eye_metrics,
            "full_assessment": full_assessment
        }
        
    except Exception as e:
        return {
            "id": position_id,
            "score": 0,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    
    finally:
        # Cleanup downloaded video
        if video_path:
            cleanup_video(video_path)


def process_interview_ui(
    video_url_1: str, question_1: str,
    video_url_2: str = "", question_2: str = "",
    video_url_3: str = "", question_3: str = "",
    video_url_4: str = "", question_4: str = "",
    video_url_5: str = "", question_5: str = "",
    progress=gr.Progress()
) -> Tuple[str, str, str, str, str, str]:
    """
    Process interviews from UI inputs.
    Returns formatted results for display.
    """
    try:
        # Collect valid interviews
        interviews = []
        video_urls = [video_url_1, video_url_2, video_url_3, video_url_4, video_url_5]
        questions = [question_1, question_2, question_3, question_4, question_5]
        
        for i, (url, question) in enumerate(zip(video_urls, questions), 1):
            if url and url.strip() and "drive.google.com" in url:
                interviews.append({
                    "positionId": i,
                    "question": question or f"Interview Question {i}",
                    "recordedVideoUrl": url.strip(),
                    "isVideoExist": True
                })
        
        if not interviews:
            return (
                "❌ Error", 
                "No valid Google Drive video URLs provided", 
                "", 
                "0", 
                "0%", 
                "{}"
            )
        
        # Process each interview
        results = []
        scores = []
        total = len(interviews)
        
        for idx, interview in enumerate(interviews):
            progress((idx) / total, f"Processing interview {idx + 1}/{total}...")
            
            result = process_single_interview(
                position_id=interview["positionId"],
                question=interview["question"],
                video_url=interview["recordedVideoUrl"],
                progress_callback=lambda msg: progress((idx + 0.5) / total, msg)
            )
            results.append(result)
            scores.append(result.get("score", 0))
        
        progress(1.0, "Completed!")
        
        # Calculate overall metrics
        avg_score = sum(scores) / len(scores) if scores else 0
        interview_percentage = (avg_score / 4) * 100
        
        # Determine decision
        if avg_score >= 3:
            decision = "✅ PASSED"
        elif avg_score >= 2:
            decision = "⚠️ REVIEW"
        else:
            decision = "❌ FAILED"
        
        # Format individual results with psychometric analysis
        result_details = []
        for r in results:
            score_stars = "⭐" * r.get("score", 0) + "☆" * (4 - r.get("score", 0))
            full_assessment = r.get("full_assessment", {})
            notes_text = full_assessment.get("notes", "")
            
            # Extract psychometric analysis components
            star = full_assessment.get("star_analysis", {})
            toulmin = full_assessment.get("toulmin_analysis", {})
            fluency = full_assessment.get("fluency_analysis", {})
            eye = full_assessment.get("eye_behavior_analysis", {})
            pronoun = full_assessment.get("pronoun_analysis", {})
            overall = full_assessment.get("overall_assessment", {})
            
            # Build STAR status
            star_status = []
            if star.get("situation_present"): star_status.append("S✓")
            else: star_status.append("S✗")
            if star.get("task_present"): star_status.append("T✓")
            else: star_status.append("T✗")
            if star.get("action_present"): star_status.append("A✓")
            else: star_status.append("A✗")
            if star.get("result_present"): star_status.append("R✓")
            else: star_status.append("R✗")
            
            detail = f"""
### Question {r['id']}
**Score:** {r.get('score', 0)}/4 {score_stars}

**Reason:** {r.get('reasoning', 'N/A')}

---
#### Psychometric Analysis

**STAR Method:** {' '.join(star_status)} | Action Specificity: {star.get('action_specificity', 'N/A')} | Result Quantified: {'Yes' if star.get('result_quantified') else 'No'}

**Argumentation (Toulmin):** Claim: {'✓' if toulmin.get('claim_present') else '✗'} | Grounds: {toulmin.get('grounds_quality', 'N/A')} | Rebuttal: {'✓' if toulmin.get('rebuttal_present') else '✗'}

**Ownership:** I-ratio: {pronoun.get('i_ratio_in_action', 'N/A')} | Level: {pronoun.get('ownership_level', 'N/A')}

**Fluency:** {fluency.get('speech_pattern', 'N/A')} | Disfluency: {fluency.get('disfluency_density', 'N/A')} | Score: {fluency.get('fluency_score', 'N/A')}/4

**Eye Behavior:** Engagement: {eye.get('engagement_level', 'N/A')} | Cognitive Aversion: {eye.get('cognitive_gaze_aversion', 'N/A')} | Integrity: {eye.get('integrity_flag', 'clean')}

---
#### 📝 Notes
{notes_text if notes_text else 'N/A'}

**Strengths:** {', '.join(overall.get('strengths', ['N/A']))}
**Areas to Improve:** {', '.join(overall.get('weaknesses', ['N/A']))}

**Transcript Preview:** {r.get('transcript', 'N/A')[:250]}{'...' if len(r.get('transcript', '')) > 250 else ''}

---"""
            result_details.append(detail)
        
        details_text = "\n".join(result_details)
        
        # Build overall notes from LLM analysis
        overall_notes_parts = []
        for r in results:
            full_assessment = r.get("full_assessment", {})
            note = full_assessment.get("notes", "")
            if note:
                overall_notes_parts.append(note)
        
        # Combine overall notes or generate summary
        if overall_notes_parts:
            overall_notes = " ".join(overall_notes_parts)
        else:
            overall_notes = f"Processed {len(interviews)} interviews. Average score: {avg_score:.1f}/4."
        
        # Build API response JSON - STRICT FORMAT per project requirements
        # Only id, score, reason in scores array
        scores_simple = []
        for r in results:
            full_assessment = r.get("full_assessment", {})
            # Reason should contain the comprehensive analysis
            reason_text = r.get("reasoning", "")
            if not reason_text or reason_text == "Assessment completed":
                reason_text = full_assessment.get("notes", "No detailed analysis available")
            
            scores_simple.append({
                "id": r["id"],
                "score": r.get("score", 0),
                "reason": reason_text
            })
        
        # Calculate interview score as sum of individual scores (percentage of max possible)
        total_score = sum(s["score"] for s in scores_simple)
        max_possible = len(scores_simple) * 4
        interview_score_pct = round((total_score / max_possible) * 100, 1) if max_possible > 0 else 0
        
        # Determine decision
        if avg_score >= 3:
            api_decision = "PASSED"
        elif avg_score >= 2:
            api_decision = "Need Human"
        else:
            api_decision = "FAILED"
        
        api_response = {
            "success": True,
            "data": {
                "pastReviews": [{
                    "assessorProfile": {
                        "id": 1,
                        "name": "AI Assessor",
                        "photoUrl": ""
                    },
                    "decision": api_decision,
                    "reviewedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "scoresOverview": {
                        "project": 0,
                        "interview": interview_score_pct,
                        "total": interview_score_pct
                    },
                    "reviewChecklistResult": {
                        "project": [],
                        "interviews": {
                            "minScore": 0,
                            "maxScore": 4,
                            "scores": scores_simple
                        }
                    },
                    "notes": overall_notes
                }]
            }
        }
        
        # Summary for UI display
        ui_notes = f"Processed {len(interviews)} interviews. Average score: {avg_score:.1f}/4"
        
        return (
            decision,
            ui_notes,
            details_text,
            f"{avg_score:.1f}/4",
            f"{interview_score_pct:.0f}%",
            json.dumps(api_response, indent=2, ensure_ascii=False)
        )
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"UI Processing Error: {traceback.format_exc()}")
        return (
            "❌ Error",
            error_msg,
            f"```\n{traceback.format_exc()}\n```",
            "0",
            "0%",
            json.dumps({"success": False, "error": str(e)})
        )


# ============================================
# API Processing Function (Backend)
# ============================================

def process_api_request(request_json: str) -> str:
    """
    API endpoint for programmatic access.
    Accepts JSON input, returns JSON output.
    """
    try:
        data = json.loads(request_json)
        
        # Extract interviews
        if "reviewChecklists" in data:
            interviews = data["reviewChecklists"].get("interviews", [])
        elif "interviews" in data:
            interviews = data["interviews"]
        elif isinstance(data, list):
            interviews = data
        else:
            return json.dumps({"success": False, "error": "Invalid input format"})
        
        if not interviews:
            return json.dumps({"success": False, "error": "No interviews found"})
        
        # Process
        results = []
        scores = []
        
        for interview in interviews:
            if not interview.get("isVideoExist", True) or not interview.get("recordedVideoUrl"):
                continue
                
            result = process_single_interview(
                position_id=interview.get("positionId", 0),
                question=interview.get("question", ""),
                video_url=interview.get("recordedVideoUrl", "")
            )
            results.append(result)
            scores.append(result.get("score", 0))
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Calculate interview score as percentage
        total_score = sum(scores)
        max_possible = len(scores) * 4
        interview_score_pct = round((total_score / max_possible) * 100, 1) if max_possible > 0 else 0
        
        # Decision based on average
        if avg_score >= 3:
            decision = "PASSED"
        elif avg_score >= 2:
            decision = "Need Human"
        else:
            decision = "FAILED"
        
        # Build scores with reason (strict format)
        scores_simple = []
        overall_notes_parts = []
        for r in results:
            full_assessment = r.get("full_assessment", {})
            reason_text = r.get("reasoning", "")
            if not reason_text or reason_text == "Assessment completed":
                reason_text = full_assessment.get("notes", "No detailed analysis available")
            
            scores_simple.append({
                "id": r["id"],
                "score": r.get("score", 0),
                "reason": reason_text
            })
            
            note = full_assessment.get("notes", "")
            if note:
                overall_notes_parts.append(note)
        
        # Overall notes
        if overall_notes_parts:
            overall_notes = " ".join(overall_notes_parts)
        else:
            overall_notes = f"Processed {len(results)} interviews. Average score: {avg_score:.1f}/4."
        
        response = {
            "success": True,
            "data": {
                "pastReviews": [{
                    "assessorProfile": {"id": 1, "name": "AI Assessor", "photoUrl": ""},
                    "decision": decision,
                    "reviewedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "scoresOverview": {
                        "project": 0,
                        "interview": interview_score_pct,
                        "total": interview_score_pct
                    },
                    "reviewChecklistResult": {
                        "project": [],
                        "interviews": {
                            "minScore": 0,
                            "maxScore": 4,
                            "scores": scores_simple
                        }
                    },
                    "notes": overall_notes
                }]
            }
        }
        
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# ============================================
# Gradio UI
# ============================================

with gr.Blocks(title="AI Interview Assessment System") as demo:
    gr.Markdown("""
    # AI-Powered Interview Assessment System
    
    Sistem ini menganalisis video wawancara menggunakan:
    - 🎤 **Faster Whisper** - Transkripsi speech-to-text
    - 👁️ **MediaPipe** - Analisis pergerakan mata & atensi
    - 🤖 **Google Gemini** - Scoring & feedback cerdas
    
    ---
    """)
    
    with gr.Tabs():
        # Tab 1: User-friendly UI
        with gr.TabItem("📝 Assessment Form"):
            gr.Markdown("### Masukkan Video Interview")
            gr.Markdown("Paste Google Drive video URL untuk pertanyaan interview.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Interview 1 (Always visible)
                    with gr.Group():
                        gr.Markdown("#### 📹 Interview 1")
                        question_1 = gr.Textbox(
                            label="Pertanyaan",
                            placeholder="Masukkan pertanyaan interview...",
                            value="Can you share any specific challenges you faced while working on certification?",
                            lines=2
                        )
                        video_url_1 = gr.Textbox(
                            label="Google Drive Video URL",
                            placeholder="https://drive.google.com/file/d/xxx/view",
                            lines=1
                        )
                    
                    # Additional interviews (collapsible)
                    with gr.Accordion("➕ Tambah Interview Lainnya (Opsional)", open=False):
                        with gr.Group():
                            gr.Markdown("#### 📹 Interview 2")
                            question_2 = gr.Textbox(
                                label="Pertanyaan",
                                placeholder="Masukkan pertanyaan interview...",
                                lines=2
                            )
                            video_url_2 = gr.Textbox(
                                label="Google Drive Video URL",
                                placeholder="https://drive.google.com/file/d/xxx/view",
                                lines=1
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 📹 Interview 3")
                            question_3 = gr.Textbox(
                                label="Pertanyaan",
                                placeholder="Masukkan pertanyaan interview...",
                                lines=2
                            )
                            video_url_3 = gr.Textbox(
                                label="Google Drive Video URL",
                                placeholder="https://drive.google.com/file/d/xxx/view",
                                lines=1
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 📹 Interview 4")
                            question_4 = gr.Textbox(
                                label="Pertanyaan",
                                placeholder="Masukkan pertanyaan interview...",
                                lines=2
                            )
                            video_url_4 = gr.Textbox(
                                label="Google Drive Video URL",
                                placeholder="https://drive.google.com/file/d/xxx/view",
                                lines=1
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 📹 Interview 5")
                            question_5 = gr.Textbox(
                                label="Pertanyaan",
                                placeholder="Masukkan pertanyaan interview...",
                                lines=2
                            )
                            video_url_5 = gr.Textbox(
                                label="Google Drive Video URL",
                                placeholder="https://drive.google.com/file/d/xxx/view",
                                lines=1
                            )
                    
                    assess_btn = gr.Button("🚀 Mulai Assessment", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    # Results display
                    gr.Markdown("### 📊 Hasil Assessment")
                    
                    decision_output = gr.Textbox(
                        label="Keputusan",
                        interactive=False,
                        lines=1
                    )
                    
                    with gr.Row():
                        avg_score_output = gr.Textbox(
                            label="Rata-rata Score",
                            interactive=False
                        )
                        percentage_output = gr.Textbox(
                            label="Persentase",
                            interactive=False
                        )
                    
                    summary_output = gr.Textbox(
                        label="Ringkasan",
                        interactive=False,
                        lines=2
                    )
                    
                    details_output = gr.Markdown(
                        label="Detail per Pertanyaan"
                    )
            
            # Hidden API response for debugging
            with gr.Accordion("🔧 API Response (Debug)", open=False):
                api_response_output = gr.Code(
                    label="JSON Response",
                    language="json",
                    lines=20
                )
            
            # Connect button
            assess_btn.click(
                fn=process_interview_ui,
                inputs=[
                    video_url_1, question_1,
                    video_url_2, question_2,
                    video_url_3, question_3,
                    video_url_4, question_4,
                    video_url_5, question_5
                ],
                outputs=[
                    decision_output,
                    summary_output,
                    details_output,
                    avg_score_output,
                    percentage_output,
                    api_response_output
                ]
            )
        
        # Tab 2: API Documentation
        with gr.TabItem("🔌 API Documentation"):
            gr.Markdown("""
            ## API Endpoint
            
            Untuk integrasi dengan sistem lain, gunakan API endpoint:
            
            ```
            POST /api/assess
            Content-Type: application/json
            ```
            
            ### Request Format
            ```json
            {
              "reviewChecklists": {
                "interviews": [
                  {
                    "positionId": 1,
                    "question": "Your interview question",
                    "isVideoExist": true,
                    "recordedVideoUrl": "https://drive.google.com/file/d/xxx/view"
                  }
                ]
              }
            }
            ```
            
            ### Response Format
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
                        "scores": [{"id": 1, "score": 3}]
                      }
                    }
                  }
                ]
              }
            }
            ```
            
            ### Python Example
            ```python
            import requests
            
            response = requests.post(
                "https://YOUR_SPACE_URL/api/assess",
                json={"data": [your_json_string]}
            )
            result = response.json()
            ```
            """)
            
            gr.Markdown("### Test API")
            api_input = gr.Code(
                label="Request JSON",
                language="json",
                value='''{
  "reviewChecklists": {
    "interviews": [
      {
        "positionId": 1,
        "question": "Can you share any specific challenges you faced?",
        "isVideoExist": true,
        "recordedVideoUrl": "https://drive.google.com/file/d/YOUR_ID/view"
      }
    ]
  }
}''',
                lines=15
            )
            api_test_btn = gr.Button("🧪 Test API", variant="secondary")
            api_output = gr.Code(label="Response JSON", language="json", lines=20)
            
            api_test_btn.click(
                fn=process_api_request,
                inputs=[api_input],
                outputs=[api_output],
                api_name="assess"
            )

# Launch
if __name__ == "__main__":
    demo.queue()  # Enable queue for long-running tasks
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )