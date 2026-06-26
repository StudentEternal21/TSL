import os
from services.audio import convert_to_wav

# ── Language prefix → 3-char ID mapping ─────────────────────────────────────
_DIALECT_PREFIX_MAP: dict[str, str] = {
    "ceb": "ceb",
    "hil": "hil",
    "ilo": "ilo",
    "war": "war",
    "kap": "kap",
}
# ────────────────────────────────────────────────────────────────────────────


def resolve_dialect_from_filename(filename: str) -> str:
    """
    Derive the 3-character language ID from an audio filename.

    The filename must begin with a recognised language prefix followed by an
    underscore (e.g. ``ceb_001.wav`` → ``"ceb"``).

    Parameters
    ----------
    filename : str
        Basename of the audio file (e.g. ``"war_20260626.wav"``).

    Returns
    -------
    str
        A valid 3-character language ID, or ``"unk"`` if the prefix is not
        recognised.
    """
    prefix = filename.split("_")[0].lower()
    return _DIALECT_PREFIX_MAP.get(prefix, "unk")


def ingest_audio_asset(
    source_audio_path: str,
    audio_type: str = "donated",
) -> tuple[str | None, str]:
    """
    Convert an audio file to the standard WAV format (16 kHz, mono) and place
    it in the appropriate subdirectory under ``data/audio_speech/``.

    This function is responsible **only** for audio preparation and storage.
    Metadata logging (transcription + RAG correction) is handled exclusively
    by ``pipeline.correct.run_correction``, which appends one row to
    ``data/metadata.csv`` after the full Whisper → RAG pipeline completes.

    Parameters
    ----------
    source_audio_path : str
        Path to the original audio file to ingest.
    audio_type : str
        Subdirectory label under ``data/audio_speech/``
        (e.g. ``"donated"``).

    Returns
    -------
    tuple[str | None, str]
        ``(output_path, message)`` on success, or ``(None, error_message)``
        on failure.
    """
    if not os.path.exists(source_audio_path):
        return None, "Error: Source audio file does not exist."

    # Determine the destination path based on audio type
    output_dir = f"data/audio_speech/{audio_type.lower()}"
    base_name = os.path.basename(source_audio_path)

    # Ensure standard .wav extension
    if not base_name.lower().endswith(".wav"):
        base_name = os.path.splitext(base_name)[0] + ".wav"

    output_path = os.path.join(output_dir, base_name)

    try:
        convert_to_wav(source_audio_path, output_path)
        return output_path, "Success: Audio asset converted and stored."
    except Exception as e:
        return None, f"Error during audio ingestion: {str(e)}"