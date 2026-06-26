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
## Setup

1. Install ollama, run `irm https://ollama.com/install.ps1 | iex` in your terminal
2. Run `ollama pull gemma4:12b snowflake-arctic-embed2` in your terminal.
3. Run `pipenv install` in your terminal.
4. Run `pipenv shell` in your terminal.
5. Run `python app.py` in your terminal.
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

