# POSO — ASR Error Correction via RAG-Enhanced Post-Processing
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

**POSO** is a hackathon project that improves Automatic Speech Recognition (ASR) accuracy by pairing OpenAI Whisper with a local Retrieval-Augmented Generation (RAG) pipeline. Whisper's raw transcription is post-processed by a RAG-backed LLM that cross-references a domain-specific text corpus to detect and correct transcription errors — particularly for specialized terminology, proper nouns, and low-resource language content. The quality of the corrected output is then measured against Whisper's baseline using standard ASR metrics.

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
│  OpenAI Whisper   │    ──►  Raw Transcript (baseline)
│  (Cloud ASR)      │
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

1. **Voice Input → OpenAI Whisper**
   - Raw audio (`.wav`, `.mp3`, etc.) is sent to the OpenAI Whisper API.
   - Whisper returns a **raw transcript** — this serves as both the input to the next stage and the **baseline** for evaluation.

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

1. Run `ollama run qwen3.5:0.8b` in your terminal.
2. Run `pipenv install` in your terminal.

---

## 🏗️ Draft Project Structure

```
TSL/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── data/                  # Text corpus and audio samples
├── src/                   # Source modules
│   ├── ingestion.py       # Whisper API integration
│   ├── augmentation.py    # RAG retriever + LLM correction
│   └── evaluation.py      # ASR evaluation metrics
└── README.md              # This file
```

