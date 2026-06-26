# POSO: ASR Error Correction via RAG-Enhanced Post-Processing
## "Ang Pag-igib sa Salita ng Bawat Pilipino."
---

## 📋 Project Information

| Field            | Details                              |
| ---------------- | ------------------------------------ |
| **Project Case** |          `Tinig sa Liwanag`          |
| **Group Name**   |           `PrimalSkidbi`             |

### 👥 Team Members

| Name                    | Role / Responsibility          |
| ----------------------- | ------------------------------ |
| `Jedrick Darren Ocenar` |  `Team Lead`                   |
| `Apacible Enegue Eoghan`|  `Developer`                   |
| `Pelayo Agatha Fei`     |  `Developer`                   |
| `Layno Ryan Reimann`    |  `Developer`                   |

---

## 🧭 Overview

**POSO** is a hackathon project that improves Automatic Speech Recognition (ASR) accuracy by pairing a local **faster-whisper** model with a local Retrieval-Augmented Generation (RAG) pipeline. Whisper's raw transcription is post-processed by a RAG-backed LLM that cross-references a domain-specific text corpus to detect and correct transcription errors — particularly for specialized terminology, proper nouns, and low-resource Philippine language content. The entire stack runs locally on consumer hardware, ensuring zero cloud costs and complete data privacy.

---

## 🔁 System Workflow

```
┌─────────────┐
│  Voice Input │
│  (Audio)     │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  faster-whisper   │    ──►  Raw Transcript (baseline)
│  (Local ASR)      │
└──────┬───────────┘
       │
       ▼
┌──────────────────────────┐
│  Local RAG LLM           │
│  ┌────────────────────┐  │
│  │  Retriever          │◄─── Text Corpus (domain knowledge)
│  │  (vector search)    │  │
│  └────────┬───────────┘  │
│           ▼              │
│  ┌────────────────────┐  │
│  │  LLM Generator     │  │
│  │  (error correction) │  │
│  └────────┬───────────┘  │
└───────────┼──────────────┘
            │
            ▼
┌───────────────────────┐
│  Corrected Transcript  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────────────────┐
│  ASR Quality Metrics              │
│  (corrected vs. Whisper baseline) │
│  • WER  • CER  • BLEU / etc.     │
└───────────────────────────────────┘
```

### Step-by-Step Breakdown

1. **Voice Input → faster-whisper**
   - Raw audio (`.wav`, `.mp3`, etc.) is transcribed locally via `faster-whisper`.
   - Whisper returns a **raw transcript**, this serves as both the input to the next stage and the **baseline** for evaluation.

2. **Whisper Output → Local RAG LLM**
   - The raw transcript is passed to a locally hosted LLM augmented with Retrieval-Augmented Generation (RAG).
   - The RAG retriever performs a **vector similarity search** against an indexed **text corpus** to find relevant context passages.

3. **RAG LLM Checks the Text Corpus**
   - The retrieved passages provide domain-specific knowledge (terminology, names, phrases) that Whisper may have misrecognised.
   - The LLM uses this context to understand what the speaker *likely* said.

4. **RAG LLM Corrects Whisper's Mistakes**
   - The LLM generates a **corrected transcript**, fixing errors such as:
     - Misspelled or substituted domain terms
     - Incorrect proper nouns
     - Contextually wrong words that are phonetically similar

5. **ASR Quality Metric Evaluation**
   - The **corrected transcript** is compared against the **raw Whisper output** (and optionally a ground-truth reference) using standard ASR metrics:

   | Metric | Description |
   | ------ | ----------- |
   | **WER** (Word Error Rate) | Percentage of words incorrectly transcribed |
   | **CER** (Character Error Rate) | Character-level error rate — useful for morphologically rich languages |
   | **BLEU** | N-gram overlap score commonly used in MT/NLP evaluation |

---
## 🛠️ Technologies Used

- **Speech Recognition (ASR)**: `faster-whisper` (Local, CPU-optimized Whisper engine)
- **Local LLM & Embedding Host**: `Ollama`
- **Correction LLM**: `Gemma 4:12b` (utilizing local Chain-of-Thought reasoning)
- **Embedding Model**: `snowflake-arctic-embed2` (used to index the dialect corpora)
- **Crowdsourcing Interface**: `Gradio` (mobile-friendly audio recording and upload portal)
- **API Endpoint**: `Flask` (for processing headless file translation/correction requests)
- **Environment Management**: `Pipenv`

---

## 🚀 Setup Instructions

### 1. Prerequisites
- **Ollama**: Download and install Ollama. On Windows PowerShell:
  ```powershell
  irm https://ollama.com/install.ps1 | iex
  ```
- Make sure to pull the correct models locally:
  ```powershell
  ollama pull gemma4:12b
  ollama pull snowflake-arctic-embed2
  ```

### 2. Python Environment Setup
1. Clone/navigate to the repository folder.
2. Initialize and install dependencies using `pipenv`:
   ```powershell
   pipenv install
   ```
3. Activate the virtual environment shell:
   ```powershell
   pipenv shell
   ```

### 3. Launching the App & API
- **To run the Crowdsourcing Web Portal (Gradio UI):**
  ```powershell
  python app.py
  ```
  Open `http://127.0.0.1:7860` in your web browser.
  
- **To run the developer integration endpoint (Flask API):**
  ```powershell
  python api.py
  ```
  Starts a REST API at `http://127.0.0.1:5000/transcribe`.

### 4. Running Integration Tests
To verify the API against local test recordings:
```powershell
python tests/test_api.py
```

---

## 🏗️ Project Structure

```text
TSL/
├── app.py                        # Main Gradio application entry point
├── Pipfile                       # Python dependencies (Pipenv)
├── data/                         # Data storage
│   ├── metadata.csv              # Master ledger for recordings & transcripts
│   ├── prompts.json              # Prompts served to the UI
│   ├── raw_text/                 # Language text corpuses (.jsonl)
│   ├── index/                    # Embedded RAG vector indices (.npy)
│   └── whisper_sound_processing/ # Saved user recordings (.wav)
├── services/                     # Core business logic
│   ├── whisper.py                # Local faster-whisper integration
│   ├── rag.py                    # RAG retrieval & LLM generation
│   └── audio.py                  # Audio processing utilities
├── pipeline/                     # Orchestration scripts
│   ├── ingest.py                 # Data ingestion logic
│   ├── augment.py                # Background noise injection
│   ├── correct.py                # RAG correction pipeline runner
│   └── evaluate.py               # ASR evaluation metrics
└── README.md                     # This file
```

---

## 🤖 AI Disclosure

Parts of this codebase, UI design, and documentation were built with the assistance of Artificial Intelligence tools to accelerate development and rapid prototyping.

