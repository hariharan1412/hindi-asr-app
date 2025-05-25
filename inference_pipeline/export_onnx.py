import nemo.collections.asr as nemo_asr
import torch # PyTorch is a dependency of NeMo
import os

# --- Configuration ---
# The name of the NeMo model
MODEL_NAME = "stt_hi_conformer_ctc_medium"
NEMO_LOCAL_PATH = f"nemo_models/{MODEL_NAME}.nemo"
ONNX_OUTPUT_PATH = f"nemo_models/{MODEL_NAME}.onnx"

print("\n--- NeMo Model Export to ONNX ---")

asr_model = None

# Step 1: Check if the .nemo file exists locally
if os.path.exists(NEMO_LOCAL_PATH):
    print(f"Found local NeMo model file: '{NEMO_LOCAL_PATH}'. Loading it...")
    try:
        # Load the model from the local path
        asr_model = nemo_asr.models.ASRModel.restore_from(
            restore_path=NEMO_LOCAL_PATH,
            map_location=torch.device('cpu') # Load to CPU for ONNX export compatibility
        )
        print("NeMo model loaded successfully from local file.")
    except Exception as e:
        print(f"Error loading local NeMo model from '{NEMO_LOCAL_PATH}': {e}")
        print("Attempting to download the model instead...")
        # If loading from local path fails, try downloading
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name=MODEL_NAME,
            map_location=torch.device('cpu')
        )
        print(f"NeMo model successfully downloaded and loaded: {MODEL_NAME}.")
else:
    # Step 2: If the .nemo file doesn't exist locally, download it
    print(f"Local NeMo model file '{NEMO_LOCAL_PATH}' not found. Downloading model: {MODEL_NAME}...")
    try:
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name=MODEL_NAME,
            map_location=torch.device('cpu') # Load to CPU for ONNX export compatibility
        )
        print(f"NeMo model successfully downloaded and loaded: {MODEL_NAME}.")
    except Exception as e:
        print(f"\nAn error occurred during model download: {e}")
        if "ConnectionError" in str(e) or "HTTP Error" in str(e) or "network" in str(e).lower():
            print("Please check your internet connection and ensure you have access to NVIDIA NGC.")
        print("Exiting. Ensure all required NeMo and PyTorch dependencies are correctly installed.")
        exit(1) # Exit if download fails

# Ensure model is set to evaluation mode
if asr_model:
    asr_model.eval()
else:
    print("Failed to load or download NeMo model. Exiting.")
    exit(1)

# Step 3: Export the loaded model to ONNX
print(f"Exporting model to ONNX at: {ONNX_OUTPUT_PATH}...")
try:
    # The export() method handles the conversion to ONNX.
    # Specify an ONNX opset version. Opset 13 is commonly used.
    asr_model.export(ONNX_OUTPUT_PATH, onnx_opset_version=13)
    print(f"Model successfully exported to {ONNX_OUTPUT_PATH}")
    print("You can now use this ONNX file with ONNX Runtime for inference.")

    # Optional: Verify the ONNX model (basic integrity check)
    import onnx
    print("\nPerforming ONNX model integrity check...")
    onnx_model = onnx.load(ONNX_OUTPUT_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model integrity check passed.")

except Exception as e:
    print(f"\nAn error occurred during ONNX export: {e}")
    print("Please ensure the NeMo model is compatible with ONNX export for the chosen opset version.")
    exit(1)