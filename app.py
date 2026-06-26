"""
POSO — Contributor Portal
A hyper-minimalist, mobile-friendly voice recording interface
for crowdsourcing Philippine language voice data.
"""

# pyrefly: ignore [missing-import]
import gradio as gr
import json
import os
import shutil
import csv
import random
from datetime import datetime
from pathlib import Path
from pipeline.correct import run_correction

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_FILE = DATA_DIR / "prompts.json"
RECORDINGS_DIR = DATA_DIR / "audio_speech"
METADATA_FILE = DATA_DIR / "metadata.csv"

LANGUAGES = ["Kapampangan","Cebuano", "Ilocano", "Hiligaynon", "Waray"]

LANGUAGE_LABELS = {
    "Kapampangan": "🌋 Kapampangan",
    "Cebuano": "🌊 Cebuano",
    "Ilocano": "🌄 Ilocano",
    "Hiligaynon": "🌺 Hiligaynon",
    "Waray": "🌿 Waray",
}

# ──────────────────────────────────────────────
# Load prompts
# ──────────────────────────────────────────────
def load_prompts():
    """Load text prompts from the JSON corpus file."""
    if PROMPTS_FILE.exists():
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback if file is missing
    return {lang: [f"Sample prompt for {lang}."] for lang in LANGUAGES}

PROMPTS = load_prompts()


# ──────────────────────────────────────────────
# Pipeline integration
# ──────────────────────────────────────────────
def submit_recording(audio_path, language, prompt_text, audio_type="donated"):
    """
    Save the recorded audio, then run the full
    Whisper → RAG correction pipeline on it.
    """
    if audio_path is None:
        return "⚠️ No recording found. Please record your voice first."

    # Place recordings under data/audio_speech/<audio_type>/
    # No language subfolder — the lang_id prefix in the filename is
    # sufficient for dialect detection in run_correction.
    lang_dir = RECORDINGS_DIR / audio_type
    lang_dir.mkdir(parents=True, exist_ok=True)

    # Map full name to 3-char ID
    lang_map = {
        "Kapampangan": "kap",
        "Cebuano": "ceb",
        "Ilocano": "ilo",
        "Hiligaynon": "hil",
        "Waray": "war"
    }
    lang_id = lang_map.get(language, "unk")

    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{lang_id}_{timestamp}.wav"
    dest_path = lang_dir / filename

    # Copy the audio file from Gradio's temp location
    shutil.copy2(audio_path, dest_path)

    # Run the Whisper → RAG correction pipeline
    # run_correction handles transcription, RAG correction, and appends
    # results (audio_path, dialect, raw_whisper_transcript, corrected_transcript)
    # to data/metadata.csv — so we read the last row back to display them.
    try:
        run_correction(str(dest_path), language=lang_id)

        # Read back the last row from metadata.csv to display results
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["audio_path"] == str(os.path.abspath(dest_path))]

        if rows:
            last = rows[-1]
            raw_transcript = last.get("raw_whisper_transcript", "")
            corrected_transcript = last.get("corrected_transcript", "")
            return (
                f"✅ Recording saved! ({filename})\n\n"
                f"**🎤 Whisper (raw):**\n{raw_transcript}\n\n"
                f"**✨ POSO (corrected):**\n{corrected_transcript}"
            )
        return f"✅ Recording saved! ({filename})"
    except Exception as exc:
        print(f"[App] Pipeline error: {exc}")
        return f"✅ Recording saved! ({filename})\n\n⚠️ Correction unavailable: {exc}"


def get_random_prompt(language):
    """Return a random prompt for the given language."""
    prompts = PROMPTS.get(language, ["No prompts available."])
    return random.choice(prompts)


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global ── */
* { font-family: 'Inter', sans-serif !important; }

.gradio-container {
    max-width: 520px !important;
    margin: 0 auto !important;
    background: #0a0a0f !important;
    min-height: 100vh;
}

body, .main, .app {
    background: #0a0a0f !important;
}

/* ── Header ── */
#app-title {
    text-align: center;
    padding: 2rem 1rem 0.25rem;
}

#app-title h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #a78bfa, #818cf8, #6366f1) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0 !important;
    letter-spacing: -0.5px;
}

#app-subtitle {
    text-align: center;
    padding: 0 1rem 1.5rem;
}

#app-subtitle p {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    font-weight: 400;
    margin: 0;
}

/* ── Language buttons ── */
#lang-btn-group {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.5rem 0;
}

.lang-btn {
    min-height: 64px !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    border-radius: 16px !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.08)) !important;
    color: #e2e8f0 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    letter-spacing: 0.5px;
}

.lang-btn:hover {
    border-color: rgba(99, 102, 241, 0.5) !important;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(139, 92, 246, 0.18)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2) !important;
}

/* ── Choose-language label ── */
#choose-label p {
    color: #cbd5e1 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.25rem;
}

/* ── Teleprompter card ── */
#prompt-card {
    background: linear-gradient(145deg, #1e1b4b, #1a1640) !important;
    border: 1px solid rgba(129, 140, 248, 0.2) !important;
    border-radius: 20px !important;
    padding: 2rem 1.5rem !important;
    margin: 0.75rem 0 !important;
    min-height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
}

#prompt-card .prose {
    text-align: center;
}

#prompt-card p, #prompt-card span {
    color: #e2e8f0 !important;
    font-size: 1.45rem !important;
    font-weight: 500 !important;
    line-height: 1.6 !important;
    letter-spacing: 0.2px;
}

/* ── Language badge ── */
#lang-badge {
    text-align: center;
    margin-bottom: 0.25rem;
}

#lang-badge span, #lang-badge p {
    color: #a78bfa !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 3px;
}

/* ── Next prompt button ── */
#next-prompt-btn {
    border-radius: 12px !important;
    background: transparent !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    min-height: 42px !important;
    transition: all 0.2s ease !important;
}

#next-prompt-btn:hover {
    border-color: rgba(148, 163, 184, 0.5) !important;
    color: #e2e8f0 !important;
}

/* ── Audio recorder ── */
#audio-recorder {
    margin: 0.75rem 0 !important;
}

#audio-recorder .record-button, #audio-recorder button {
    border-radius: 14px !important;
}

/* ── Submit button ── */
#submit-btn {
    min-height: 56px !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

#submit-btn:hover {
    background: linear-gradient(135deg, #818cf8, #a78bfa) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.35) !important;
}

/* ── Back button ── */
#back-btn {
    min-height: 40px !important;
    border-radius: 12px !important;
    background: transparent !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

#back-btn:hover {
    border-color: rgba(148, 163, 184, 0.4) !important;
    color: #e2e8f0 !important;
}

/* ── Status message ── */
#status-msg {
    text-align: center;
    margin-top: 0.5rem;
}

#status-msg p, #status-msg span {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
    margin: 0.75rem 0;
}

/* ── Footer ── */
footer { display: none !important; }

/* ── Hide extra Gradio chrome ── */
.gradio-container .contain { gap: 0 !important; }
"""


# ──────────────────────────────────────────────
# Build the Gradio UI
# ──────────────────────────────────────────────
with gr.Blocks(title="POSO — Contributor Portal") as app:

    # ── Hidden state ──
    selected_language = gr.State("")
    current_prompt = gr.State("")

    # ════════════════════════════════════════════
    # SCREEN 1 — Language Selection
    # ════════════════════════════════════════════
    with gr.Column(visible=True, elem_id="screen-select") as screen_select:
        gr.Markdown("# POSO", elem_id="app-title")
        gr.Markdown("Ang Pag-igib sa Salita ng Bawat Pilipino", elem_id="app-subtitle")

        gr.HTML("<hr class='divider'>")

        gr.Markdown("Choose your language", elem_id="choose-label")

        with gr.Column(elem_id="lang-btn-group"):
            btn_kapampangan = gr.Button("🌋  Kapampangan", elem_classes="lang-btn")
            btn_cebuano = gr.Button("🌊  Cebuano", elem_classes="lang-btn")
            btn_ilocano = gr.Button("🌄  Ilocano", elem_classes="lang-btn")
            btn_hiligaynon = gr.Button("🌺  Hiligaynon", elem_classes="lang-btn")
            btn_waray = gr.Button("🌿  Waray", elem_classes="lang-btn")

    # ════════════════════════════════════════════
    # SCREEN 2 — Record & Submit
    # ════════════════════════════════════════════
    with gr.Column(visible=False, elem_id="screen-record") as screen_record:
        back_btn = gr.Button("← Back", elem_id="back-btn")

        lang_badge = gr.Markdown("", elem_id="lang-badge")

        prompt_display = gr.Markdown("", elem_id="prompt-card")

        next_btn = gr.Button("🔀  Next Prompt", elem_id="next-prompt-btn")

        gr.HTML("<hr class='divider'>")

        audio_input = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="🎙️ Record or upload audio",
            elem_id="audio-recorder",
        )

        submit_btn = gr.Button("Submit Recording", elem_id="submit-btn")

        status_msg = gr.Markdown("", elem_id="status-msg")

    # ──────────────────────────────────────────
    # Event handlers
    # ──────────────────────────────────────────

    def select_language(lang):
        """Switch to the recording screen with a random prompt."""
        prompt = get_random_prompt(lang)
        return (
            gr.update(visible=False),       # hide selection screen
            gr.update(visible=True),        # show recording screen
            lang,                           # update state
            prompt,                         # update state
            f"{LANGUAGE_LABELS.get(lang, lang)}",  # badge text
            prompt,                         # prompt card text
            None,                           # clear audio
            "",                             # clear status
        )

    def go_back():
        """Return to the language selection screen."""
        return (
            gr.update(visible=True),        # show selection screen
            gr.update(visible=False),       # hide recording screen
            "",                             # clear language state
            "",                             # clear prompt state
            "",                             # clear badge
            "",                             # clear prompt card
            None,                           # clear audio
            "",                             # clear status
        )

    def next_prompt(language):
        """Load a new random prompt for the current language."""
        prompt = get_random_prompt(language)
        return prompt, prompt, None, ""

    def on_submit(audio, language, prompt_text):
        """Handle the submit action."""
        result = submit_recording(audio, language, prompt_text)
        if result.startswith("✅"):
            # Also get a fresh prompt after successful submission
            new_prompt = get_random_prompt(language)
            return result, new_prompt, new_prompt, None
        return result, gr.update(), gr.update(), gr.update()

    # ── Wire up language buttons ──
    lang_outputs = [
        screen_select, screen_record,
        selected_language, current_prompt,
        lang_badge, prompt_display,
        audio_input, status_msg,
    ]

    btn_kapampangan.click(
        fn=lambda: select_language("Kapampangan"),
        outputs=lang_outputs,
    )
    btn_cebuano.click(
        fn=lambda: select_language("Cebuano"),
        outputs=lang_outputs,
    )
    btn_ilocano.click(
        fn=lambda: select_language("Ilocano"),
        outputs=lang_outputs,
    )
    btn_hiligaynon.click(
        fn=lambda: select_language("Hiligaynon"),
        outputs=lang_outputs,
    )
    btn_waray.click(
        fn=lambda: select_language("Waray"),
        outputs=lang_outputs,
    )

    # ── Back button ──
    back_btn.click(
        fn=go_back,
        outputs=lang_outputs,
    )

    # ── Next prompt ──
    next_btn.click(
        fn=next_prompt,
        inputs=[selected_language],
        outputs=[current_prompt, prompt_display, audio_input, status_msg],
    )

    # ── Submit ──
    submit_btn.click(
        fn=on_submit,
        inputs=[audio_input, selected_language, current_prompt],
        outputs=[status_msg, current_prompt, prompt_display, audio_input],
    )


# ──────────────────────────────────────────────
# Launch
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(),
    )
