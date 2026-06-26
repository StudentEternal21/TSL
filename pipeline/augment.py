import os
from services.audio import load_audio, save_audio, resample, normalize

def inject_filipino_noise(clean_audio_path, noise_type, output_path="data/whisper_sound_processing", gain_db=-15):
    """
    Injects local background noise into a clean audio file.
    """
    if not os.path.exists(clean_audio_path):
        return None, "Error: Clean audio file does not exist."
    
    # Standard fallback noise path
    noise_path = f"data/noise_profiles/{noise_type.lower()}.wav"

    try: 
        # Load the clean audio
        clean_audio = load_audio(clean_audio_path)

        if not os.path.exists(noise_path):
            # If the specified noise profile does not exist, apply a default gain to the clean audio to simulate noise
            augmented_audio = clean_audio.apply_gain(-2)
            reason = f"Noise profile '{noise_type}' not found. Applied default gain of -2 dB."
        else:
            noise_audio = load_audio(noise_path)
            
            if len(noise_audio) < len(clean_audio):
                # Loop the noise to match the length of the clean audio
                noise_audio = noise_audio * (len(clean_audio) // len(noise_audio) + 1)
            noise_audio = noise_audio[:len(clean_audio)]

            # Mix the clean audio with the noise audio at the specified gain
            augmented_audio = clean_audio.overlay(noise_audio.apply_gain(gain_db))
            reason = f"Injected noise from '{noise_type}' with gain {gain_db} dB."
        
        # Export the augmented audio
        base_name = os.path.basename(clean_audio_path)
        output_file_path = os.path.join(output_path, f"noisy_{noise_type.lower()}_{base_name}")

        # Ensure output matches baseline training dimensions
        augmented_audio = resample(augmented_audio, 16000)
        augmented_audio = normalize(augmented_audio, 1)
        save_audio(augmented_audio, output_file_path)
        
        return output_file_path, reason
    except Exception as e:
        return None, f"Error during audio augmentation: {str(e)}"