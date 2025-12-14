# Panduan AI Interview Assessment System

> Dokumentasi untuk instalasi lokal dan penggunaan API

---

## Daftar Isi

1. [Instalasi Lokal](#-instalasi-lokal)
2. [Penggunaan API](#-penggunaan-api)

---

## Instalasi Lokal

### Prasyarat

| Software | Keterangan |
|----------|------------|
| Python 3.10+ | [Download Python](https://www.python.org/downloads/) |
| FFmpeg | Untuk processing audio/video |
| Git | Untuk clone repository |
| Gemini API Key | Gratis dari [Google AI Studio](https://aistudio.google.com/app/apikey) |

### Langkah 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Interview-Assessment-System.git
cd AI-Interview-Assessment-System
```

### Langkah 2: Buat Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Langkah 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Langkah 4: Install FFmpeg

**Windows (Chocolatey):**
```powershell
choco install ffmpeg
```

**Windows (Winget):**
```powershell
winget install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### Langkah 5: Set API Key

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY = "your_api_key_here"
```

**Linux/macOS:**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Langkah 6: Jalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di: `http://localhost:7860`

---

## 🔌 Penggunaan API

### Endpoint

```
POST /api/assess
Content-Type: application/json
```

### Format Request

```json
{
  "reviewChecklists": {
    "interviews": [
      {
        "positionId": 1,
        "question": "Ceritakan pengalaman mengatasi tantangan di pekerjaan sebelumnya",
        "isVideoExist": true,
        "recordedVideoUrl": "https://drive.google.com/file/d/FILE_ID/view"
      }
    ]
  }
}
```

### Contoh Kode Python

```python
import requests
import json

# URL aplikasi (local atau HF Spaces)
BASE_URL = "http://localhost:7860"  # atau "https://your-space.hf.space"

# Data request
data = {
    "reviewChecklists": {
        "interviews": [
            {
                "positionId": 1,
                "question": "Ceritakan pengalaman mengatasi tantangan di pekerjaan",
                "isVideoExist": True,
                "recordedVideoUrl": "https://drive.google.com/file/d/YOUR_FILE_ID/view"
            }
        ]
    }
}

# Kirim request
response = requests.post(
    f"{BASE_URL}/api/assess",
    json={"data": [json.dumps(data)]}
)

# Tampilkan hasil
result = response.json()
print(json.dumps(result, indent=2, ensure_ascii=False))
```

### Format Response

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
        "reviewedAt": "2025-12-06 15:00:00",
        "scoresOverview": {
          "project": 0,
          "interview": 75,
          "total": 75
        },
        "reviewChecklistResult": {
          "interviews": {
            "minScore": 0,
            "maxScore": 4,
            "scores": [
              {
                "id": 1,
                "score": 3,
                "reason": "Kandidat memberikan jawaban terstruktur..."
              }
            ]
          }
        },
        "notes": "Kandidat menunjukkan kemampuan komunikasi yang baik"
      }
    ]
  }
}
```

### Penjelasan Response

| Field | Keterangan |
|-------|------------|
| `decision` | `PASSED` / `Need Human` / `FAILED` |
| `scoresOverview.interview` | Persentase skor interview (0-100) |
| `scores[].score` | Skor per pertanyaan (0-4) |
| `scores[].reason` | Penjelasan detail penilaian |
| `notes` | Ringkasan keseluruhan |

### Skala Penilaian

| Score | Label | Kriteria |
|-------|-------|----------|
| 0 | Very Poor | Tidak relevan atau ada indikasi kecurangan |
| 1 | Poor | Minimal relevansi, tidak terstruktur |
| 2 | Average | Menjawab tapi kurang mendalam |
| 3 | Good | Terstruktur dengan contoh spesifik |
| 4 | Excellent | Komprehensif dan data-driven |

### Keputusan Otomatis

| Rata-rata Score | Keputusan |
|-----------------|-----------|
| ≥ 3.0 | ✅ PASSED |
| 2.0 - 2.9 | ⚠️ Need Human (Perlu review) |
| < 2.0 | ❌ FAILED |

---

## 🎙️ Model Speech-to-Text (Fine-Tuned)

Model STT yang digunakan adalah **Whisper Large V3 Turbo** yang telah di-fine-tune untuk meningkatkan akurasi transkripsi, terutama untuk Bahasa Indonesia dan English dengan aksen Indonesia.

### Detail Fine-Tuning

| Parameter | Nilai |
|-----------|-------|
| **Base Model** | `openai/whisper-large-v3-turbo` |
| **Fine-Tuned Model** | `Dafisns/whisper-turbo-multilingual-fleurs` |
| **Metode** | LoRA (Low-Rank Adaptation) |
| **Trainable Parameters** | 27.8M (3.33% dari total) |
| **Epochs** | 2 |

### Dataset Training

| Dataset | Bahasa | Train | Test |
|---------|--------|-------|------|
| Google Fleurs | English (en_us) | 2,600 | 320 |
| Google Fleurs | Indonesian (id_id) | 2,550 | 320 |
| Common Voice | English | 5,200 | 600 |
| Common Voice | Indonesian | 4,500 | 500 |
| **EdACC** | **English (aksen Indonesia)** | 172 | 43 |
| **Total** | - | **15,022** | **1,783** |

> **EdACC (Edinburgh Accented Corpus)** adalah dataset English dengan aksen Indonesia, sangat relevan untuk konteks wawancara di Indonesia.

### Hasil Evaluasi (WER - Word Error Rate)

| Bahasa | Dataset Evaluasi | WER |
|--------|------------------|-----|
| **English** | Fleurs + Common Voice + EdACC | **9.09%** |
| **Indonesian** | Fleurs + Common Voice | **6.97%** |

*Semakin rendah WER, semakin akurat transkripsi. WER < 10% memenuhi hasil yang diharapkan.*

---

## Troubleshooting

| Error | Solusi |
|-------|--------|
| `GEMINI_API_KEY not set` | Set environment variable dengan API key |
| `Could not open video` | Pastikan video Google Drive di-share "Anyone with the link" |
| `FFmpeg not found` | Install FFmpeg dan restart terminal |

---
