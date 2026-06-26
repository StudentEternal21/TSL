import os
import pandas as pd
from pydub import AudioSegment

METADATA_FILE = "data/metadata.csv"

def initialize_ledger():
    """
    Initializes the metadata ledger for audio files.
    If the metadata file already exists, it will be overwritten.
    """
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(METADATA_FILE):
        df = pd.DataFrame(columns=["audio_path", "dialect", "transcript_clean", "audio_type"])
        df.to_csv(METADATA_FILE, index=False)

def ingest_audio_asset(source_audio_path, dialect, transcript, audio_type="donation"):
    """
    Ingests a new audio asset into the system and updates the metadata ledger.
    """
    initialize_ledger()

    if not os.path.exists(source_audio_path):
        return None, "Error: Source audio file does not exist."
    
    # Determine the destination path based on audio type
    output_dir = f"data/audio_speech/{audio_type.lower()}"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(source_audio_path)
    # Ensure standard extension and format (16kHz, Mono, WAV)
    if not base_name.lower().endswith(".wav"):
        base_name = os.path.splitext(base_name)[0] + ".wav"
    
    output_path = os.path.join(output_dir, base_name)

    try:
        # Load and normalize the audio track
        audio = AudioSegment.from_file(source_audio_path)

        # Enforce 16kHz sample rate and mono channel
        normalized_audio = audio.set_frame_rate(16000).set_channels(1)

        # Export file as compressed WAV format
        normalized_audio.export(output_path, format="wav")

        # Log asset ingestion in the metadata ledger
        df = pd.read_csv(METADATA_FILE)

        # Prevent appending duplicate paths
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