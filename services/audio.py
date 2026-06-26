import os
from pydub import AudioSegment

def load_audio(file_path):
    """Safely loads an audio file into a pydub AudioSegment object."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    return AudioSegment.from_file(file_path)

def save_audio(audio_segment, output_path, format="wav"):
    """Exports an audio segment to a given output path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio_segment.export(output_path, format=format)
    return output_path

def resample(audio_segment, target_frame_rate=16000):
    """Sets the frame rate of the audio segment (Standard: 16kHz for ML)."""
    return audio_segment.set_frame_rate(target_frame_rate)

def normalize(audio_segment, target_channels=1):
    """Sets the audio channels (Standard: 1 for Mono)."""
    return audio_segment.set_channels(target_channels)

def convert_to_wav(file_path, output_path):
    """Convert any audio source directly to a standard WAV target."""
    audio = load_audio(file_path)
    audio = resample(audio, 16000)
    audio = normalize(audio, 1)
    return save_audio(audio, output_path, format="wav")