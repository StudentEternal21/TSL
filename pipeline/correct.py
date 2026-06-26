from services.whisper import transcribe_file
from services.rag import correct_transcript

if __name__ == "__main__":
    import sys

    audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/whisper_sound_processing/sample.wav"

    raw_transcript = transcribe_file(audio_path)
    print(f"[Whisper] Raw transcript:\n{raw_transcript}\n")

    corrected, context = correct_transcript(raw_transcript)
    print(f"[RAG] Corrected transcript:\n{corrected}\n")
    print(f"[RAG] Retrieved context:\n{context}")
