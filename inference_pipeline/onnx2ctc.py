import onnxruntime as ort
import soundfile as sf
import numpy as np
import os
import torch 
import nemo.collections.asr as nemo_asr

# --- Configuration ---
MODEL_NAME = "stt_hi_conformer_ctc_medium"
NEMO_MODEL_PATH = f"nemo_models/{MODEL_NAME}.nemo"
ONNX_MODEL_PATH = f"nemo_models/{MODEL_NAME}.onnx"
AUDIO_FILE_PATH = "hindi_converted.wav"


# --- Audio File Check and Dummy Creation (for testing) ---
# For actual transcription, you MUST provide a valid Hindi WAV file.
if not os.path.exists(AUDIO_FILE_PATH):
    print(f"Audio file '{AUDIO_FILE_PATH}' not found. Attempting to create a dummy WAV for testing...")
    samplerate = 16000 
    duration = 3 
    frequency = 800 
    t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2. * np.pi * frequency * t)
    sf.write(AUDIO_FILE_PATH, audio_data.astype(np.float32), samplerate, subtype='PCM_16')
    print(f"Dummy audio file created at '{AUDIO_FILE_PATH}'.")
    print("NOTE: This dummy audio will NOT produce meaningful Hindi transcription.")
    print("Please replace it with a real, valid Hindi WAV audio file for actual use!")
else:
    print(f"Using provided audio file: '{AUDIO_FILE_PATH}'. Please ensure it is a valid WAV.")


# --- Check if ONNX model exists ---
if not os.path.exists(ONNX_MODEL_PATH):
    print(f"Error: ONNX model file not found at '{ONNX_MODEL_PATH}'")
    print("Please ensure you have successfully exported your NeMo model to ONNX.")
    exit()

## Step 1: Load the ONNX Model
print(f"Loading ONNX model from: {ONNX_MODEL_PATH}...")
# Create an ONNX Runtime inference session.
# Uses CPUExecutionProvider for CPU-only inference.
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
print("ONNX model loaded successfully.")

# Get input and output names from the ONNX model
input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]

print(f"ONNX Model Inputs: {input_names}")
print(f"ONNX Model Outputs: {output_names}")

print("Loading NeMo model to retrieve tokenizer and audio preprocessor (this might take a moment if not cached)...")

if not os.path.exists(NEMO_MODEL_PATH):
    print(f"Error: NeMo model file not found at '{NEMO_MODEL_PATH}'")
    print("Please ensure your .nemo file is in the same directory.")
    exit()

asr_model_for_tokenizer_and_preprocessor = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    restore_path=NEMO_MODEL_PATH,
    map_location=torch.device('cpu')
)

tokenizer = asr_model_for_tokenizer_and_preprocessor.tokenizer
print(f"NeMo tokenizer loaded successfully.")

audio_preprocessor = asr_model_for_tokenizer_and_preprocessor.preprocessor
print("NeMo audio preprocessor loaded successfully.")


## Step 3: Load and Preprocess Audio 
print(f"Loading audio file: {AUDIO_FILE_PATH}...")
audio_signal, sample_rate = sf.read(AUDIO_FILE_PATH) 
# Ensure audio is mono 
if audio_signal.ndim > 1:
    audio_signal = audio_signal[:, 0]
# Convert to float32, which is the expected data type for most neural networks
audio_signal = audio_signal.astype(np.float32)

# Resample to 16kHz if necessary
if sample_rate != 16000:
    print(f"Warning: Audio sample rate is {sample_rate}Hz. Resampling to 16000Hz...")
    from resampy import resample # resampy is required for efficient resampling
    audio_signal = resample(audio_signal, sample_rate, 16000)
    sample_rate = 16000
print(f"Raw audio loaded. Sample rate: {sample_rate}Hz, Duration: {len(audio_signal)/sample_rate:.2f}s")

# Convert raw audio to features using the NeMo preprocessor
# The preprocessor expects a PyTorch tensor, so convert numpy array to torch tensor
audio_signal_torch = torch.from_numpy(audio_signal).unsqueeze(0) # Add batch dimension
audio_signal_length_torch = torch.tensor([len(audio_signal)], dtype=torch.int64)

# Apply the preprocessor
processed_audio_features, processed_audio_features_length = audio_preprocessor(
    input_signal=audio_signal_torch,
    length=audio_signal_length_torch
)

# Convert processed features back to numpy for ONNX Runtime
processed_audio_features_np = processed_audio_features.squeeze(0).cpu().numpy() # Remove batch dim, move to CPU, convert to numpy
processed_audio_features_length_np = processed_audio_features_length.cpu().numpy()

print(f"Audio preprocessed to features. Feature shape: {processed_audio_features_np.shape}, Feature length: {processed_audio_features_length_np[0]}")


## Step 4: Prepare Inputs for ONNX Runtime
# The ONNX model now expects the preprocessed features.
# Expected shape for audio_signal: [batch_size, channels, time_steps]
# For features, channels = number of Mel bins (e.g., 80), time_steps = length of feature sequence.
# We need to add a batch dimension to the features.
input_features_batch = processed_audio_features_np[np.newaxis, :, :] # Reshape to [1, channels, time_steps]
input_features_length_batch = processed_audio_features_length_np 

# Create the input dictionary for ONNX Runtime.
# Ensure the keys match the input names of your ONNX model as identified in Step 1.
onnx_inputs = {
    "audio_signal": input_features_batch, # Now passing the preprocessed features
    "length": input_features_length_batch
}

## Step 5: Run ONNX Inference
print("Running ONNX inference...")
onnx_outputs = session.run(output_names, onnx_inputs)
# The ONNX model typically outputs logits as the first element.
logits = onnx_outputs[0]
print("ONNX inference complete.")

## Step 6: Decode the CTC Logits

# The logits from the ONNX model are typically of shape [Batch, Time, Num_Tokens].
logits_np = logits[0] # Shape now [Time, Num_Tokens]

print(f"Shape of logits_np before argmax: {logits_np.shape}")
num_output_tokens_from_onnx = logits_np.shape[-1]
print(f"Number of output tokens from ONNX model: {num_output_tokens_from_onnx}")

# Dynamically determine blank_id based on ONNX output dimension.
if num_output_tokens_from_onnx == tokenizer.vocab_size + 1:
    blank_id = num_output_tokens_from_onnx - 1 
    print(f"Adjusted blank ID based on ONNX output dimension: {blank_id}")
else:
    blank_id = tokenizer.vocab_size - 1
    print(f"Blank ID remains assumed (tokenizer.vocab_size - 1): {blank_id}")

# Step 6: Decode the CTC Logits
print("Decoding transcription...")
# Perform greedy CTC decoding manually:
# 1. Get the most probable token ID at each timestep (argmax).
predicted_ids_np = np.argmax(logits_np, axis=-1) # This is an array of numpy integers

# Add debug prints for predicted_ids range
print(f"Max predicted ID after argmax: {np.max(predicted_ids_np)}")
print(f"Min predicted ID after argmax: {np.min(predicted_ids_np)}")

# 2. Apply CTC decoding rules:
#    a. Remove consecutive duplicate token IDs.
#    b. Remove blank tokens (using the blank_id obtained from the tokenizer).
decoded_ids = []
for i, token_id_numpy in enumerate(predicted_ids_np):
    if token_id_numpy == blank_id: # blank_id is a Python int, comparison is fine
         continue
    # predicted_ids_np[i-1] is also a numpy integer. Comparison is fine.
    if i > 0 and token_id_numpy == predicted_ids_np[i-1]:
         continue
    # CRITICAL CHANGE: Convert numpy integer to Python int before appending
    decoded_ids.append(int(token_id_numpy))

print("Decoded ids : ", decoded_ids) # This list now contains Python integers
# 3. Convert the list of decoded token IDs back into text using the NeMo tokenizer.
transcribed_text = tokenizer.ids_to_text(decoded_ids)

print("\n--- Transcription Result ---")
print(f"Transcribed Text: {transcribed_text}")
print("----------------------------")