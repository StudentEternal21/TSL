import os
from faster_whisper import WhisperModel

# ── Tuneable constants ──────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "large-v3"          # faster-whisper model size
WHISPER_DEVICE     = "cuda"              # "cuda" or "cpu"
WHISPER_COMPUTE    = "float16"           # "float16" (GPU) or "int8" (CPU)
RAW_SOUND_DIR      = "data/raw_sound"   # default audio input folder
# ────────────────────────────────────────────────────────────────────────────

# Supported audio extensions
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".webm"}


def _resolve_path(relative: str) -> str:
    """Resolve a project-root-relative path to an absolute path."""
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    return os.path.normpath(os.path.join(project_root, relative))


def transcribe_file(
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
        Absolute or project-root-relative path to the audio file.
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
    """
    if not os.path.isabs(audio_path):
        audio_path = _resolve_path(audio_path)

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

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


def transcribe_folder(
    folder_path: str        = RAW_SOUND_DIR,
    model_size: str         = WHISPER_MODEL_SIZE,
    device: str             = WHISPER_DEVICE,
    compute_type: str       = WHISPER_COMPUTE,
    language: str | None    = None,
) -> dict[str, str]:
    """
    Transcribe every supported audio file inside *folder_path*.

    The model is loaded once and reused across all files for efficiency.

    Parameters
    ----------
    folder_path : str
        Absolute or project-root-relative path to the audio folder.
        Defaults to ``data/raw_sound``.
    model_size : str
        faster-whisper model size tag.
    device : str
        Inference device — ``"cuda"`` or ``"cpu"``.
    compute_type : str
        Quantisation type.
    language : str | None
        BCP-47 language code to force, or ``None`` for auto-detection.

    Returns
    -------
    dict[str, str]
        Mapping of ``filename → transcript`` for every audio file found.
        Files that fail are mapped to an error string prefixed with
        ``"ERROR: "``.
    """
    if not os.path.isabs(folder_path):
        folder_path = _resolve_path(folder_path)

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Audio folder not found: {folder_path}")

    audio_files = sorted(
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in _AUDIO_EXTENSIONS
    )

    if not audio_files:
        print(f"[Whisper] No supported audio files found in: {folder_path}")
        return {}

    print(f"[Whisper] Loading model '{model_size}' on {device} ({compute_type}) …")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    results: dict[str, str] = {}
    total = len(audio_files)

    for i, filename in enumerate(audio_files, start=1):
        audio_path = os.path.join(folder_path, filename)
        print(f"[Whisper] ({i}/{total}) Transcribing: {filename}")
        try:
            segments, info = model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            transcript = " ".join(seg.text.strip() for seg in segments)
            print(f"[Whisper]   └─ lang={info.language} "
                  f"({info.language_probability:.2f})  "
                  f"chars={len(transcript)}")
            results[filename] = transcript
        except Exception as exc:
            error_msg = f"ERROR: {exc}"
            print(f"[Whisper]   └─ FAILED — {exc}")
            results[filename] = error_msg

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transcripts = transcribe_folder()
    print("\n── Transcripts ──────────────────────────────────────────────────")
    for fname, text in transcripts.items():
        print(f"\n{fname}:\n  {text}")
