import nemo.collections.asr as nemo_asr
import torchaudio
import torch
import os
import numpy as np
import soundfile as sf
import warnings # Import warnings module

# --- Configuration ---
# The Hindi Conformer-CTC Medium model from NVIDIA NGC
MODEL_NAME = "stt_hi_conformer_ctc_medium"
AUDIO_FILE_TO_TEST = "hindi.wav"
EXPECTED_SAMPLE_RATE = 16000 # Expected sample rate for the NeMo model

# --- Helper Function to Create Dummy Audio ---
def create_dummy_audio_file(filepath: str, duration_seconds: int = 7, sample_rate: int = EXPECTED_SAMPLE_RATE):
    """
    Generates a dummy WAV file with a simple sine wave, suitable for initial ASR testing.
    The duration is set to 7 seconds, within the 5-10 second range.
    """
    print(f"Creating dummy audio file: {filepath} ({duration_seconds}s at {sample_rate}Hz)...")
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False)
    amplitude = 0.5
    frequency = 440  # A4 note
    data = amplitude * np.sin(2 * np.pi * frequency * t)
    try:
        sf.write(filepath, data.astype(np.float32), sample_rate)
        print("Dummy audio file created successfully.")
    except Exception as e:
        print(f"Error creating dummy audio file: {e}")
        print("Please ensure 'soundfile' is installed (`pip install soundfile`).")

# --- Main Native NeMo Model Testing Logic ---
if __name__ == "__main__":
    print("\n--- Native NeMo ASR Model Local Testing ---")

    if not os.path.exists(AUDIO_FILE_TO_TEST):
        print(f"Test audio file '{AUDIO_FILE_TO_TEST}' not found.")
        create_dummy_audio_file(filepath=AUDIO_FILE_TO_TEST)
        print("Note: Transcribing dummy audio will not yield meaningful Hindi text.")
    else:
        print(f"Using existing audio file for test: {AUDIO_FILE_TO_TEST}")

    print(f"\nLoading NeMo ASR model: {MODEL_NAME}...")

    try:
        # Load the pre-trained ASR model from NeMo.
        # This will download the model weights if not already cached.
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=MODEL_NAME)
        print("NeMo model loaded successfully!")

        # Move model to GPU if available for faster inference
        if torch.cuda.is_available():
            asr_model = asr_model.cuda()
            print("Model moved to GPU for inference.")
        else:
            print("CUDA not available, running on CPU.")

        # Set the model to evaluation mode (important for inference)
        asr_model.eval()

        print(f"\nLoading audio file: {AUDIO_FILE_TO_TEST}")
        # Load audio file using torchaudio
        waveform, sample_rate = torchaudio.load(AUDIO_FILE_TO_TEST)

        # --- Preprocessing for NeMo Model ---
        # Resample if not at the expected sample rate (16kHz for this model)
        if sample_rate != EXPECTED_SAMPLE_RATE:
            print(f"Resampling audio from {sample_rate}Hz to {EXPECTED_SAMPLE_RATE}Hz...")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=EXPECTED_SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = EXPECTED_SAMPLE_RATE

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            print("Converting stereo audio to mono...")
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Check audio duration 
        duration = waveform.shape[1] / EXPECTED_SAMPLE_RATE 
        print(f"Audio duration: {duration:.2f} seconds.")
        if not (5 <= duration <= 10):
            print("Warning: Audio duration is outside the recommended 5-10 second range for optimal performance.")
            print("For the FastAPI endpoint, audio will be clipped if too long.")

        # Perform transcription using the native NeMo model's built-in method.
        # This method handles internal preprocessing and decoding.
        print("\nPerforming transcription using the native NeMo model...")
        hypotheses = asr_model.transcribe([AUDIO_FILE_TO_TEST]) 
        
        if hypotheses:
            print("\n--- Transcription Result ---")
            print(f"Transcribed Text: \"{hypotheses[0]}\"")
        else:
            print("No transcription found.")

    except Exception as e:
        print(f"\nAn error occurred during model loading or transcription: {e}")
        if "Failed to download" in str(e) or "connection" in str(e):
            print("Please check your internet connection or NeMo NGC access.")
        elif "No such file or directory" in str(e):
            print(f"Ensure '{AUDIO_FILE_TO_TEST}' exists or the path is correct.")
