"""
Flask API — Audio Transcription & Correction
POST /transcribe  →  saves audio, runs Whisper→RAG pipeline, returns corrected transcript
"""

import csv
import os

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

# ── Configuration ─────────────────────────────────────────────────────────────
_PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
_UPLOAD_DIR     = os.path.join(_PROJECT_ROOT, "data", "whisper_sound_processing")
_METADATA_CSV   = os.path.join(_PROJECT_ROOT, "data", "metadata.csv")

_VALID_DIALECTS     = {"ceb", "ilo", "hil", "war", "kap"}
_ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".webm"}

os.makedirs(_UPLOAD_DIR, exist_ok=True)

# ── Import pipeline (deferred so Flask starts fast even if deps are slow) ─────
from pipeline.correct import run_correction  # noqa: E402

app = Flask(__name__)


# ── Helper ────────────────────────────────────────────────────────────────────

def _last_csv_row() -> dict | None:
    """Return the last data row from metadata.csv as a dict, or None."""
    if not os.path.isfile(_METADATA_CSV) or os.path.getsize(_METADATA_CSV) == 0:
        return None
    with open(_METADATA_CSV, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else None


# ── Route ─────────────────────────────────────────────────────────────────────

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accept an audio file and a dialect code, run the correction pipeline,
    and return the corrected transcript.

    Form fields
    -----------
    audio   : file   — the audio file to transcribe
    dialect : str    — language ID: one of ceb | ilo | hil | war | kap
    """
    # ── Validate dialect ──────────────────────────────────────────────────────
    dialect = request.form.get("dialect", "").strip().lower()
    if not dialect:
        return jsonify({"error": "Missing form field 'dialect'."}), 400
    if dialect not in _VALID_DIALECTS:
        return jsonify({
            "error": f"Invalid dialect '{dialect}'. Valid options: {sorted(_VALID_DIALECTS)}"
        }), 400

    # ── Validate file ─────────────────────────────────────────────────────────
    if "audio" not in request.files:
        return jsonify({"error": "Missing file field 'audio'."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(audio_file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported audio format '{ext}'. "
                     f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
        }), 400

    # ── Save to whisper_sound_processing ──────────────────────────────────────
    save_path = os.path.join(_UPLOAD_DIR, filename)
    audio_file.save(save_path)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        run_correction(save_path, language=dialect)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Pipeline error: {exc}"}), 500

    # ── Read result from CSV ──────────────────────────────────────────────────
    row = _last_csv_row()
    if row is None:
        return jsonify({"error": "Pipeline completed but CSV row was not written."}), 500

    return jsonify({
        "dialect":               row.get("dialect"),
        "audio_path":            row.get("audio_path"),
        "raw_whisper_transcript": row.get("raw_whisper_transcript"),
        "corrected_transcript":  row.get("corrected_transcript"),
    }), 200


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
