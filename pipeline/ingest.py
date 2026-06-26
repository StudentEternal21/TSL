import os
import pandas as pd
from services.audio import save_audio, convert_to_wav

METADATA_FILE = "data/metadata.csv"

def initialize_ledger():
    """
    Initializes the metadata ledger for audio files.
    """
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(METADATA_FILE):
        df = pd.DataFrame(columns=["audio_path", "dialect", "transcript_clean", "audio_type"])
        df.to_csv(METADATA_FILE, index=False)

def resolve_dialect_from_filename(filename):
    """Maps filename prefixes to formal dialect names."""
    mapping = {
        "ceb": "Cebuano",
        "hil": "Hiligaynon",
        "ilo": "Ilocano",
        "war": "Waray"
    }
    # Get the prefix (e.g., 'ceb' from 'ceb_001.wav')
    prefix = filename.split('_')[0].lower()
    return mapping.get(prefix, "Unknown")

def ingest_audio_asset(source_audio_path, dialect, transcript, audio_type="donation"):
    """
    Ingests a new audio asset into the system and updates the metadata ledger.
    """
    initialize_ledger()

    if not os.path.exists(source_audio_path):
        return None, "Error: Source audio file does not exist."
    
    # Determine the destination path based on audio type
    output_dir = f"data/audio_speech/{audio_type.lower()}"
    base_name = os.path.basename(source_audio_path)\

    # Ensure standard extension and format (16kHz, Mono, WAV)
    if not base_name.lower().endswith(".wav"):
        base_name = os.path.splitext(base_name)[0] + ".wav"
    
    output_path = os.path.join(output_dir, base_name)

    try:
        convert_to_wav(source_audio_path, output_path)

        # Log asset ingestion in the metadata ledger
        df = pd.read_csv(METADATA_FILE)
        dialect = resolve_dialect_from_filename(os.path.basename(source_audio_path))
        if output_path not in df["audio_path"].values:
            new_entry = {
                "audio_path": output_path,
                "dialect": dialect.lower(),
                "transcript_clean": transcript.strip(),
                "audio_type": audio_type
            }
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_csv(METADATA_FILE, index=False)
        
        return output_path, "Success: Audio asset ingested and metadata updated."
    except Exception as e:
        return None, f"Error during audio ingestion: {str(e)}"