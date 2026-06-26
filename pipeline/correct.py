from services.whisper import transcribe_file
from services.rag import correct_transcript


def run_correction(audio_path: str) -> str:
    """
    Transcribe an audio file with Whisper then correct the transcript via RAG.

    Parameters
    ----------
    audio_path : str
        Absolute or project-root-relative path to the audio file.

    Returns
    -------
    str
        The RAG-corrected transcript.

    Raises
    ------
    ValueError
        If the corrected transcript returned by the RAG pipeline is empty.
    """
    raw_transcript = transcribe_file(audio_path)
    print(f"[Whisper] Raw transcript:\n{raw_transcript}\n")

    corrected, context = correct_transcript(raw_transcript)
    print(f"[RAG] Retrieved context:\n{context}\n")

    if not corrected or not corrected.strip():
        raise ValueError(
            "[Correction] RAG pipeline returned an empty transcript. "
            "Check the LLM/embedding service and corpus path."
        )

    print(f"[RAG] Corrected transcript:\n{corrected}\n")
    return corrected


if __name__ == "__main__":
    import sys

    audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/whisper_sound_processing/sample.wav"
    result = run_correction(audio_path)
    print(f"[Done] Final corrected transcript:\n{result}")
