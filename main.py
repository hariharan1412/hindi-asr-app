import onnxruntime as ort
import soundfile as sf
import numpy as np
import os
import torch
import nemo.collections.asr as nemo_asr
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse # FileResponse removed as not used directly by this snippet
from fastapi.staticfiles import StaticFiles
from typing import Optional, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import io
import resampy 
from pydub import AudioSegment 
import shutil 


# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
ONNX_MODEL_PATH = "nemo_models/stt_hi_conformer_ctc_medium.onnx"
NEMO_MODEL_PATH = "nemo_models/stt_hi_conformer_ctc_medium.nemo"


# Audio constraints for input validation and processing
# Model processes 5-10 seconds of audio. Files longer than 10s will be clipped.
MIN_AUDIO_DURATION_SEC = 5
MAX_AUDIO_DURATION_SEC = 10 # Audio will be clipped to this duration if longer
EXPECTED_SAMPLE_RATE = 16000 # Standard sample rate for ASR models

# --- Global Variables for Model & Session (Singleton Pattern) ---
onnx_session: Optional[ort.InferenceSession] = None
tokenizer: Optional[Any] = None
audio_preprocessor: Optional[torch.nn.Module] = None

executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# Initialize FastAPI application
app = FastAPI(
    title="ASR Inference API (NVIDIA NeMo ONNX)",
    description="An asynchronous API for transcribing Hindi audio. Processes 5-10s WAV audio (16kHz). Longer files are clipped.",
    version="1.1.0", 
)

# Serve the static HTML file for the UI
if not os.path.exists("static"):
    os.makedirs("static", exist_ok=True)
    logger.info("Created 'static' directory. Place your 'upload_form.html' inside it.")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """
    Loads the ONNX model and NeMo model during startup. Verifies FFmpeg is in the system PATH.

    Raises a RuntimeError if either model file is not found or if an error occurs during loading.
    """

    global onnx_session, tokenizer, audio_preprocessor
    logger.info("Starting up FastAPI application...")

    # --- FFmpeg Check (Optional, but good practice if pydub is used extensively) ---
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logger.warning("!!! FFmpeg not found in system PATH. Pydub might need it   !!!")
        logger.warning("!!! for some audio conversions. Install FFmpeg if issues arise. !!!")
        logger.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        logger.info(f"FFmpeg found at: {ffmpeg_exe}")


    if not os.path.exists(ONNX_MODEL_PATH):
        logger.error(f"Error: ONNX model file not found at '{ONNX_MODEL_PATH}'")
        raise RuntimeError(f"ONNX model file not found at '{ONNX_MODEL_PATH}'")

    if not os.path.exists(NEMO_MODEL_PATH):
        logger.error(f"Error: NeMo model file not found at '{NEMO_MODEL_PATH}'")
        raise RuntimeError(f"NeMo model file not found at '{NEMO_MODEL_PATH}'")

    try:
        logger.info(f"Loading ONNX model from: {ONNX_MODEL_PATH}...")
        onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        logger.info("ONNX model loaded successfully.")

        logger.info(f"Loading NeMo model from {NEMO_MODEL_PATH} to retrieve tokenizer and audio preprocessor...")
        asr_model_for_tokenizer_and_preprocessor = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            restore_path=NEMO_MODEL_PATH,
            map_location=torch.device('cpu')
        )
        tokenizer = asr_model_for_tokenizer_and_preprocessor.tokenizer
        audio_preprocessor = asr_model_for_tokenizer_and_preprocessor.preprocessor
        logger.info("NeMo tokenizer and audio preprocessor loaded successfully.")

    except Exception as e:
        logger.error(f"Failed to load models during startup: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load models: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources when the application shuts down.

    This event is triggered automatically by FastAPI when the application is shutting down.
    It is important to shut down the ThreadPoolExecutor to avoid process hangs.
    """
    logger.info("Shutting down FastAPI application...")
    executor.shutdown(wait=True)
    logger.info("ThreadPoolExecutor shut down.")

def _transcribe_audio_sync(audio_data: np.ndarray, sample_rate: int) -> str:
    """
    Sync transcription function that takes raw audio data and sample rate as input.
    Validates and resamples audio (if necessary) to 16kHz, then preprocesses it using the NeMo ASR model's preprocessor.
    The ONNX model is then used to transcribe the audio, and the CTC output is decoded to text using the NeMo tokenizer.
    This function is called by the main endpoint handler, which wraps it in an async function.

    Args:
        audio_data (np.ndarray): Raw audio data (mono, float32 or int16/32)
        sample_rate (int): Sample rate of the audio data

    Returns:
        str: The transcribed text

    Raises:
        HTTPException: If the input audio is invalid or if an error occurs during transcription
        RuntimeError: If the models are not loaded or if an unexpected error occurs during transcription
    """

    global onnx_session, tokenizer, audio_preprocessor

    if onnx_session is None or tokenizer is None or audio_preprocessor is None:
        raise RuntimeError("Models are not loaded. Server might not have started correctly.")

    try:
        # Resampling if necessary 
        if sample_rate != EXPECTED_SAMPLE_RATE:
            logger.warning(f"Audio sample rate is {sample_rate}Hz. Resampling to {EXPECTED_SAMPLE_RATE}Hz...")
            try:
                # Ensure audio_data is float before resampling
                if not np.issubdtype(audio_data.dtype, np.floating):
                     audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max if np.issubdtype(audio_data.dtype, np.integer) else audio_data.astype(np.float32)

                audio_data = resampy.resample(audio_data, sr_orig=float(sample_rate), sr_new=float(EXPECTED_SAMPLE_RATE), filter='kaiser_best')
            except ImportError:
                logger.error("resampy not found. Cannot resample audio.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Audio not at {EXPECTED_SAMPLE_RATE}Hz and resampy is not installed."
                )
            except Exception as e:
                logger.error(f"Error during resampling: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error during audio resampling: {e}"
                )
        
        if audio_data.dtype != np.float32: # Ensure float32 for model
            audio_data = audio_data.astype(np.float32)

        audio_signal_torch = torch.from_numpy(audio_data).unsqueeze(0)
        audio_signal_length_torch = torch.tensor([len(audio_data)], dtype=torch.int64)

        processed_audio_features, processed_audio_features_length = audio_preprocessor(
            input_signal=audio_signal_torch,
            length=audio_signal_length_torch
        )
        
        input_features_batch = processed_audio_features.cpu().numpy()
        input_features_length_batch = processed_audio_features_length.cpu().numpy()

        onnx_input_names = [inp.name for inp in onnx_session.get_inputs()]
        onnx_output_names = [out.name for out in onnx_session.get_outputs()]

        onnx_inputs = {
            onnx_input_names[0]: input_features_batch,
            onnx_input_names[1]: input_features_length_batch
        }

        onnx_outputs = onnx_session.run(onnx_output_names, onnx_inputs)
        logits = onnx_outputs[0][0]

        predicted_ids_np = np.argmax(logits, axis=-1)

        blank_id: int = -1
        # More robust blank_id determination
        if hasattr(tokenizer, 'blank_id') and isinstance(tokenizer.blank_id, int):
            blank_id = tokenizer.blank_id
        elif hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer.tokenizer, 'pad_id') and isinstance(tokenizer.tokenizer.pad_id, int):
            blank_id = tokenizer.tokenizer.pad_id
        elif hasattr(tokenizer, 'vocab_size') and isinstance(tokenizer.vocab_size, int):
            if logits.shape[-1] == (tokenizer.vocab_size + 1): # Common for CTC models from NeMo
                blank_id = tokenizer.vocab_size
            else: # Fallback, less ideal
                blank_id = logits.shape[-1] - 1 # Assuming blank is the last class if others fail
                logger.warning(f"Using a guessed blank_id: {blank_id}. Transcription might be affected if incorrect.")
        else: # Should not happen if tokenizer is loaded correctly
            blank_id = logits.shape[-1] - 1
            logger.error(f"Could not reliably determine blank_id. Defaulting to {blank_id}. This is likely incorrect.")


        if blank_id == -1: # Should be caught by above, but as a safeguard
            raise RuntimeError("Unable to determine CTC blank_id for decoding. Please check tokenizer properties.")

        decoded_ids = []
        last_id = -1
        for token_id_numpy in predicted_ids_np:
            token_id_int = int(token_id_numpy)
            if token_id_int != last_id and token_id_int != blank_id:
                decoded_ids.append(token_id_int)
            last_id = token_id_int

        transcribed_text = tokenizer.ids_to_text(decoded_ids)
        return transcribed_text

    except HTTPException: # Re-raise HTTPExceptions to be handled by FastAPI
        raise
    except RuntimeError as e: # Catch known RuntimeErrors (like model loading issues)
        logger.error(f"Runtime error during sync transcription: {e}", exc_info=True)
        raise e # Re-raise to be caught by the main endpoint handler
    except Exception as e:
        logger.error(f"Unexpected error during sync transcription: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error during transcription: {e}")

@app.post("/transcribe", summary="Transcribe audio to text", response_model=dict)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description=f"Audio file (WAV format, 16kHz preferred). Processes {MIN_AUDIO_DURATION_SEC}-{MAX_AUDIO_DURATION_SEC}s. Longer files clipped to {MAX_AUDIO_DURATION_SEC}s.")
):
    """
    Endpoint to transcribe an audio file to text using the NeMo ASR model.

    Supports audio files in WAV format with a sample rate of 16kHz (though other sample rates may work).
    Processes audio files with durations between 1 and 11 seconds. Longer files will be clipped to the first 11 seconds.

    Returns a JSON response with a single key "transcription" containing the transcribed text.
    If an audio file requires processing (e.g., clipping or resampling), a "note" key will be included in the response with a descriptive message.
    """
    logger.info(f"Received transcription request for file: {audio_file.filename}, content type: {audio_file.content_type}")

    # Allowing common WAV content types
    if audio_file.content_type not in ["audio/wav", "audio/x-wav", "audio/wave"]:
        logger.warning(f"Received content type {audio_file.content_type}. Attempting to process as WAV, but pydub/ffmpeg might be required if not standard WAV.")

    audio_bytes = await audio_file.read()
    
    audio_data = None
    sample_rate = None

    try:
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32', always_2d=False)
        logger.info(f"Successfully read audio with soundfile. Sample rate: {sample_rate}Hz, Shape: {audio_data.shape}")
    except sf.LibsndfileError as e:
        logger.warning(f"Soundfile failed to read audio directly: {e}. Attempting conversion with pydub/FFmpeg...")
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            converted_audio_bytes_io = io.BytesIO()
            # Export to ensure WAV format, 16kHz, mono, 16-bit PCM
            audio_segment.export(
                converted_audio_bytes_io, 
                format="wav", 
                codec="pcm_s16le", 
                parameters=["-ac", "1", "-ar", str(EXPECTED_SAMPLE_RATE)]
            )
            converted_audio_bytes_io.seek(0)
            audio_data, sample_rate = sf.read(converted_audio_bytes_io, dtype='float32', always_2d=False)
            logger.info(f"Successfully converted and read audio with pydub. New SR: {sample_rate}Hz, Shape: {audio_data.shape}")
        except FileNotFoundError as fnf_error: 
            logger.error(f"Pydub/FFmpeg execution failed: {fnf_error}. FFmpeg might be missing or not in PATH.", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Audio conversion failed: FFmpeg not found or error during execution. Original error: {fnf_error}"
            )
        except Exception as convert_e:
            logger.error(f"Failed to convert audio using pydub: {convert_e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not process audio. On-the-fly conversion failed. Error: {convert_e}"
            )
    except Exception as e:
        logger.error(f"Unexpected error during initial audio reading: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during audio processing: {e}"
        )

    if audio_data is None: 
        logger.critical("Audio data is None after all processing attempts. This should not happen.")
        raise HTTPException(status_code=500, detail="Internal error: Failed to load audio data.")

    if audio_data.ndim > 1: # Ensure mono
        audio_data = audio_data[:, 0]
        logger.info("Converted stereo audio to mono.")

    original_duration = len(audio_data) / sample_rate
    logger.info(f"Original audio duration: {original_duration:.2f}s for file: {audio_file.filename}")

    processing_note = None
    processed_duration = original_duration

    # Validate minimum duration
    if original_duration < MIN_AUDIO_DURATION_SEC:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio duration is {original_duration:.2f}s, which is less than the minimum required {MIN_AUDIO_DURATION_SEC}s."
        )

    # Clip if duration exceeds MAX_AUDIO_DURATION_SEC
    if original_duration > MAX_AUDIO_DURATION_SEC:
        logger.info(f"Original audio duration {original_duration:.2f}s exceeds maximum processing duration {MAX_AUDIO_DURATION_SEC}s. Clipping to {MAX_AUDIO_DURATION_SEC}s.")
        num_samples_to_keep = int(MAX_AUDIO_DURATION_SEC * sample_rate)
        audio_data = audio_data[:num_samples_to_keep] # audio_data is now clipped
        processed_duration = MAX_AUDIO_DURATION_SEC # Duration of data being sent for transcription
        processing_note = f"Input audio was {original_duration:.2f}s long. It has been clipped to the first {MAX_AUDIO_DURATION_SEC:.1f}s for processing as per requirements."
        logger.info(f"Audio clipped. Effective duration for processing: {processed_duration:.2f}s.")


    logger.info(f"Processing audio file '{audio_file.filename}' - Effective Duration: {processed_duration:.2f}s, Sample Rate: {sample_rate}Hz (target for model is {EXPECTED_SAMPLE_RATE}Hz)")

    try:
        transcribed_text = await asyncio.get_running_loop().run_in_executor(
            executor, _transcribe_audio_sync, audio_data, sample_rate 
        )

        logger.info(f"Successfully transcribed audio file: {audio_file.filename}. Transcription: '{transcribed_text}'")
        
        response_content = {"transcription": transcribed_text}
        if processing_note:
            response_content["note"] = processing_note
        
        return JSONResponse(content=response_content)

    except HTTPException as e:
        logger.error(f"Re-raising HTTPException from transcription process: Status {e.status_code}, Detail: {e.detail}", exc_info=False) # exc_info=False as it's already logged in _transcribe_audio_sync or here
        raise e
    except RuntimeError as e: 
        logger.error(f"Runtime error during transcription task execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred during transcription: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during transcription task execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal server error occurred: {e}"
        )

@app.get("/", response_class=HTMLResponse, summary="Serve ASR Web UI")
async def serve_upload_form():
    """
    Serves the HTML form for audio file upload from the static directory.

    If the file is not found, a fallback error HTML page is served with a 500 status code.
    """
    html_file_path = os.path.join("static", "upload_form.html")
    try:
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logger.error(f"UI file '{html_file_path}' not found.")
        fallback_html = """
        <!DOCTYPE html><html><head><title>ASR Demo - UI Error</title></head>
        <body><h1>Error: UI File Not Found</h1><p>The HTML file (expected at <code>static/upload_form.html</code>) was not found.</p></body></html>
        """
        return HTMLResponse(content=fallback_html, status_code=500)
