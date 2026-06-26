import csv
import os

from services.whisper import transcribe
from services.rag import correct_transcript, CORPUS_MAP

# Path to the metadata CSV, relative to the project root
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_METADATA_CSV = os.path.join(_PROJECT_ROOT, "data", "metadata.csv")


def run_correction(audio_path: str, language: str | None = None) -> None:
    """
    Transcribe a single audio file, correct the transcript via RAG, and
    append one row to ``data/metadata.csv``.

    The language corpus is selected from the first three characters of the
    audio file's basename by default (e.g. ``war_...wav`` → ``"war"``).  Pass
    *language* explicitly to override this behaviour (useful when the filename
    does not carry a language prefix).

    Parameters
    ----------
    audio_path : str
        Absolute or project-root-relative path to the audio file.
    language : str | None
        Language ID override (``"ceb"``, ``"ilo"``, ``"hil"``, ``"war"``,
        ``"kap"``).  When ``None`` the ID is derived from the first three
        characters of the filename.

    Raises
    ------
    ValueError
        If the resolved language ID is not recognised.
    """
    filename = os.path.basename(audio_path)

    if language is None:
        language = filename[:3].lower()

    if language not in CORPUS_MAP:
        raise ValueError(
            f"[Correction] Could not determine language from filename '{filename}'. "
            f"Resolved language ID '{language}' is not valid. "
            f"Valid options: {list(CORPUS_MAP.keys())}"
        )

    print(f"[Correction] Audio   : {filename}")
    print(f"[Correction] Language: {language}")

    # Stage 1 — Transcribe
    raw_transcript = transcribe(audio_path)
    print(f"[Whisper] Raw transcript:\n{raw_transcript}\n")

    # Stage 2 — RAG correction
    corrected, context = correct_transcript(raw_transcript, language=language)
    print(f"[RAG] Retrieved context:\n{context}\n")

    if not corrected or not corrected.strip():
        print(
            "[Correction] Warning: RAG pipeline returned an empty transcript. "
            "Falling back to raw Whisper transcript."
        )
        corrected = raw_transcript
    else:
        print(f"[RAG] Corrected transcript:\n{corrected}\n")

    # Stage 3 — Append to metadata.csv
    row = {
        "audio_path":            os.path.abspath(audio_path),
        "dialect":               language,
        "raw_whisper_transcript": raw_transcript,
        "corrected_transcript":  corrected,
    }

    file_exists = os.path.isfile(_METADATA_CSV)
    with open(_METADATA_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["audio_path", "dialect", "raw_whisper_transcript", "corrected_transcript"],
        )
        if not file_exists or os.path.getsize(_METADATA_CSV) == 0:
            writer.writeheader()
        writer.writerow(row)

    print(f"[Correction] Row appended to {_METADATA_CSV}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.correct <audio_path>")
        sys.exit(1)

    run_correction(sys.argv[1])
