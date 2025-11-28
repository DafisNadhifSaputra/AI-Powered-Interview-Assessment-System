"""
Gemini Scoring Module
Uses Google Gemini LLM to score interview responses
Based on psychometric framework from I/O Psychology research
"""

import os
import json
import re
from typing import Dict, Optional, Tuple
from google import genai
from google.genai import types


def get_gemini_client() -> genai.Client:
    """
    Initialize Gemini client.
    Uses GEMINI_API_KEY environment variable.
    
    Returns:
        Gemini client instance
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please set it with: set GEMINI_API_KEY=your_api_key")
    
    return genai.Client(api_key=api_key)


def create_assessment_prompt(
    question: str,
    transcript: str,
    eye_metrics: Dict,
    position_id: int
) -> str:
    """
    Create prompt for interview assessment based on psychometric framework.
    Uses STAR method, Toulmin argumentation, and validated I/O psychology metrics.
    
    Args:
        question: Interview question
        transcript: Transcribed answer
        eye_metrics: Eye tracking analysis results
        position_id: Position/question ID
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert Industrial/Organizational (I/O) Psychologist specializing in structured interview assessment.
Your evaluation must follow evidence-based psychometric principles with validated predictive validity.

## Interview Context
- **Question ID**: {position_id}
- **Question**: {question}

## Candidate's Response (Transcribed)
{transcript if transcript else "[No speech detected or transcription failed]"}

## Eye Movement Data (Interpreted with Cognitive Load Theory)
- **Eye Contact Percentage**: {eye_metrics.get('eye_contact_percentage', 0):.1f}%
- **Gaze Stability**: {eye_metrics.get('gaze_stability', 0):.1f}%
- **Attention Score**: {eye_metrics.get('attention_score', 0):.1f}%
- **Looking Away Percentage**: {eye_metrics.get('looking_away_percentage', 0):.1f}%

**IMPORTANT - Cognitive Gaze Aversion (CGA) Interpretation:**
- Looking away WHILE THINKING is NORMAL and POSITIVE - it indicates deep cognitive processing
- Constant staring without breaks may indicate memorized/scripted answers
- Only flag as suspicious if: repetitive horizontal eye movement (reading pattern) or consistently looking at fixed off-screen point

---

## EVALUATION FRAMEWORK

### 1. STAR Method Analysis (Behavioral Interview Structure)
Analyze the response structure. For behavioral questions, optimal distribution is:
- **Situation (S)**: 10-15% - Context setting
- **Task (T)**: 10-15% - Specific responsibility  
- **Action (A)**: 50-60% - MOST CRITICAL - Concrete steps taken by candidate
- **Result (R)**: 15-20% - Quantifiable outcomes or clear conclusions

### 2. Toulmin Argumentation Analysis (Critical Thinking)
Identify these elements in the response:
- **Claim**: Main assertion or conclusion
- **Grounds/Data**: Evidence supporting the claim (specific facts, numbers, examples)
- **Warrant**: Logical connection between data and claim
- **Rebuttal**: Acknowledgment of counter-arguments (indicates mature thinking)

### 3. Pronoun Analysis (Individual Contribution)
In the ACTION segment specifically:
- Calculate ratio of "I/me/my" vs "we/us/our"
- Ratio < 0.2 in Action = candidate may be taking credit for team work
- Balanced ratio = good teamwork with clear individual contribution

### 4. Fluency & Authenticity Assessment
- **Disfluency Density**: Count filler words (um, uh, like, you know) per 100 words
  - < 3 per 100 words = Natural fluency
  - 3-5 per 100 words = Slight nervousness, still acceptable
  - > 5 per 100 words = High anxiety or unprepared
- **Speech Pattern**: Natural pauses are GOOD (authentic thinking), robotic perfection is SUSPICIOUS

### 5. Eye Behavior Interpretation (Evidence-Based)
- **Gaze Aversion during thinking**: POSITIVE indicator of cognitive processing
- **Consistent eye contact while speaking key points**: Good engagement
- **Horizontal scanning pattern**: Potential reading from notes/script
- **Fixed off-screen focus**: Potential second monitor or external help

---

## SCORING RUBRIC (Behaviorally Anchored Rating Scale - BARS)

**Score 0 - Very Poor:**
- No relevant content addressing the question
- Clear signs of cheating (reading, external help)
- Cannot identify any STAR components
- No logical structure whatsoever

**Score 1 - Poor:**
- Minimal relevance to question
- Missing most STAR components (only Situation, no Action/Result)
- Weak or no supporting evidence
- High disfluency (>7 fillers per 100 words)

**Score 2 - Average:**
- Addresses question but lacks depth
- Has Situation and some Action, but weak/missing Result
- General statements without specific examples
- Moderate disfluency, acceptable nervousness

**Score 3 - Good:**
- Clearly addresses the question with relevant content
- Complete STAR structure with clear Action and Result
- Provides specific examples with some data/metrics
- Natural speech pattern with good fluency
- Appropriate eye contact pattern

**Score 4 - Excellent:**
- Comprehensive, well-structured response
- Strong STAR with quantified Results
- Toulmin elements present (Claim + Data + Warrant)
- Shows nuanced thinking (acknowledges trade-offs/limitations)
- High "I" ratio in Action segment showing ownership
- Confident, natural delivery

---

## Required Output (JSON ONLY)

Provide your assessment in this exact JSON format:
{{
    "score": <0-4>,
    "reason": "<1-2 sentence justification for the score>",
    
    "star_analysis": {{
        "situation_present": <true|false>,
        "task_present": <true|false>,
        "action_present": <true|false>,
        "action_specificity": "<vague|moderate|specific>",
        "result_present": <true|false>,
        "result_quantified": <true|false>,
        "star_score": <0-4>,
        "distribution_assessment": "<explanation of S-T-A-R balance>"
    }},
    
    "toulmin_analysis": {{
        "claim_present": <true|false>,
        "grounds_present": <true|false>,
        "grounds_quality": "<anecdotal|factual|data-driven>",
        "warrant_present": <true|false>,
        "rebuttal_present": <true|false>,
        "argumentation_score": <0-4>,
        "reasoning": "<explanation of argument quality>"
    }},
    
    "pronoun_analysis": {{
        "i_ratio_in_action": <0.0-1.0>,
        "ownership_level": "<low|moderate|high>",
        "assessment": "<explanation of individual vs team contribution clarity>"
    }},
    
    "fluency_analysis": {{
        "disfluency_density": "<low|moderate|high>",
        "filler_words_detected": "<none|few|many>",
        "speech_pattern": "<natural|rehearsed|nervous|robotic>",
        "fluency_score": <0-4>,
        "assessment": "<explanation of speech fluency and authenticity>"
    }},
    
    "eye_behavior_analysis": {{
        "cognitive_gaze_aversion": "<appropriate|excessive|absent>",
        "reading_pattern_detected": <true|false>,
        "engagement_level": "<low|moderate|high>",
        "integrity_flag": "<clean|warning|suspicious>",
        "interpretation": "<evidence-based interpretation of eye behavior>"
    }},
    
    "overall_assessment": {{
        "strengths": ["<strength 1>", "<strength 2>"],
        "weaknesses": ["<weakness 1>", "<weakness 2>"],
        "confidence_level": "<LOW|MEDIUM|HIGH>",
        "authenticity_score": <0-4>
    }},
    
    "notes": "<DETAILED NOTES FOR INTERVIEWER: Summarize 1) Content quality and STAR compliance, 2) Argumentation strength, 3) Speech fluency observations, 4) Eye behavior interpretation, 5) Any red flags or exceptional qualities. Write 3-5 sentences.>",
    
    "improvement_suggestions": "<Specific, actionable advice for the candidate to improve>"
}}
"""
    return prompt


def parse_gemini_response(response_text: str) -> Dict:
    """
    Parse Gemini response to extract JSON.
    
    Args:
        response_text: Raw response from Gemini
    
    Returns:
        Parsed dictionary
    """
    # Try to extract JSON from response
    try:
        # First, try direct JSON parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block in response
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{[\s\S]*\}'
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    # If all parsing fails, return default structure
    return {
        "score": 2,
        "confidence_level": "LOW",
        "reasoning": f"Could not parse response: {response_text[:200]}...",
        "error": "Response parsing failed"
    }


def assess_interview(
    question: str,
    transcript: str,
    eye_metrics: Dict,
    position_id: int,
    model: str = "gemini-2.0-flash"
) -> Tuple[int, str, Dict]:
    """
    Assess an interview response using Gemini.
    
    Args:
        question: Interview question
        transcript: Transcribed answer
        eye_metrics: Eye tracking results
        position_id: Position/question ID
        model: Gemini model to use
    
    Returns:
        Tuple of (score, reasoning, full_assessment)
    """
    client = get_gemini_client()
    
    # Create assessment prompt
    prompt = create_assessment_prompt(question, transcript, eye_metrics, position_id)
    
    try:
        # Generate response
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,  # Lower temperature for consistent psychometric scoring
                top_p=0.95,
                max_output_tokens=2048  # Increased for comprehensive analysis
            )
        )
        
        # Parse response
        response_text = response.text
        assessment = parse_gemini_response(response_text)
        
        # Extract score and reasoning
        score = assessment.get("score", 2)
        score = max(0, min(4, int(score)))  # Clamp to 0-4
        
        # Get reason (brief) and notes (detailed observations)
        reason = assessment.get("reason", assessment.get("reasoning", "Assessment completed"))
        notes = assessment.get("notes", "")
        
        # Calculate weighted final score based on psychometric framework
        # Verbal (STAR + Toulmin) = 75%, Visual (Eye) = 15%, Fluency = 10%
        star_score = assessment.get("star_analysis", {}).get("star_score", score)
        toulmin_score = assessment.get("toulmin_analysis", {}).get("argumentation_score", score)
        fluency_score = assessment.get("fluency_analysis", {}).get("fluency_score", score)
        
        # Weighted calculation (Late Fusion approach)
        verbal_score = (star_score + toulmin_score) / 2
        weighted_score = (0.75 * verbal_score) + (0.15 * score) + (0.10 * fluency_score)
        final_score = max(0, min(4, round(weighted_score)))
        
        # Check for integrity veto
        integrity_flag = assessment.get("eye_behavior_analysis", {}).get("integrity_flag", "clean")
        if integrity_flag == "suspicious":
            final_score = max(0, final_score - 2)  # Heavy penalty for integrity issues
            notes += " [INTEGRITY WARNING: Suspicious behavior detected]"
        
        # Add computed fields to assessment
        assessment["notes"] = notes
        assessment["reason"] = reason
        assessment["weighted_score"] = round(weighted_score, 2)
        assessment["final_score"] = final_score
        
        return final_score, reason, assessment
        
    except Exception as e:
        error_msg = f"Gemini API error: {str(e)}"
        return 2, error_msg, {"error": error_msg, "score": 2}


def batch_assess_interviews(
    interviews: list,
    transcripts: Dict[int, str],
    eye_metrics: Dict[int, Dict],
    model: str = "gemini-2.0-flash"
) -> list:
    """
    Assess multiple interviews.
    
    Args:
        interviews: List of interview data with positionId and question
        transcripts: Dict mapping positionId to transcript
        eye_metrics: Dict mapping positionId to eye metrics
        model: Gemini model to use
    
    Returns:
        List of assessment results
    """
    results = []
    
    for interview in interviews:
        position_id = interview.get("positionId", 0)
        question = interview.get("question", "")
        
        transcript = transcripts.get(position_id, "")
        metrics = eye_metrics.get(position_id, {})
        
        score, reasoning, full_assessment = assess_interview(
            question=question,
            transcript=transcript,
            eye_metrics=metrics,
            position_id=position_id,
            model=model
        )
        
        results.append({
            "id": position_id,
            "score": score,
            "reasoning": reasoning,
            "full_assessment": full_assessment
        })
    
    return results


if __name__ == "__main__":
    print("Gemini Scoring module loaded")
    print("Set GEMINI_API_KEY environment variable before use")
