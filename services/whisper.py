import os
from faster_whisper import WhisperModel

# ── Tuneable constants ──────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "large-v3"          # faster-whisper model size
WHISPER_DEVICE     = "cpu"               # "cuda" (NVIDIA GPU) or "cpu"
WHISPER_COMPUTE    = "int8"              # "float16" (CUDA) or "int8" (CPU)
# ────────────────────────────────────────────────────────────────────────────

# Supported audio extensions
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".webm"}


def _load_model(
    model_size: str,
    device: str,
    compute_type: str,
) -> WhisperModel:
    """
    Load a WhisperModel, automatically falling back to CPU/int8 if the
    requested device fails (e.g. CUDA driver version mismatch).
    """
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"[Whisper] Model loaded on {device} ({compute_type})")
        return model
    except RuntimeError as exc:
        if device == "cpu":
            raise   # already on CPU — nothing to fall back to
        print(f"[Whisper] {device.upper()} unavailable ({exc}). "
              f"Falling back to CPU/int8 …")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("[Whisper] Model loaded on cpu (int8)")
        return model


def _resolve_path(relative: str) -> str:
    """Resolve a project-root-relative path to an absolute path."""
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    return os.path.normpath(os.path.join(project_root, relative))


def transcribe(
    audio_path: str,
    model_size: str = WHISPER_MODEL_SIZE,
    device: str     = WHISPER_DEVICE,
    compute_type: str = WHISPER_COMPUTE,
    language: str | None = None,
) -> str:
    """
    Transcribe a single audio file using faster-whisper.

    Parameters
    ----------
    audio_path : str
        Absolute or project-root-relative path to exactly one audio file.
        Must be a file, not a directory.
    model_size : str
        faster-whisper model size tag (e.g. ``"large-v3"``, ``"medium"``).
    device : str
        Inference device — ``"cuda"`` or ``"cpu"``.
    compute_type : str
        Quantisation type — ``"float16"`` for GPU, ``"int8"`` for CPU.
    language : str | None
        BCP-47 language code to force (e.g. ``"tl"``, ``"ceb"``).
        ``None`` lets Whisper auto-detect the language.

    Returns
    -------
    str
        The full transcribed text as a single string.

    Raises
    ------
    ValueError
        If ``audio_path`` is a directory or has an unsupported extension.
    FileNotFoundError
        If ``audio_path`` does not exist.
    """
    if not os.path.isabs(audio_path):
        audio_path = _resolve_path(audio_path)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"[Whisper] Audio file not found: {audio_path}")

    if os.path.isdir(audio_path):
        raise ValueError(
            f"[Whisper] Expected a single audio file, got a directory: {audio_path}"
        )

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in _AUDIO_EXTENSIONS:
        raise ValueError(
            f"[Whisper] Unsupported audio format '{ext}'. "
            f"Supported: {', '.join(sorted(_AUDIO_EXTENSIONS))}"
        )

    model = _load_model(model_size, device, compute_type)

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,           # suppress silent/non-speech segments
        vad_parameters={"min_silence_duration_ms": 500},
    )

    print(f"[Whisper] Detected language: {info.language} "
          f"(probability {info.language_probability:.2f})")

    transcript = " ".join(seg.text.strip() for seg in segments)
    return transcript


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/whisper_sound_processing/sample.wav"
    result = transcribe(audio_path)
    print(f"\n── Transcript ───────────────────────────────────────────────────\n{result}")
