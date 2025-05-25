# Use a lightweight Python base image
FROM python:3.10-slim-buster


# Set the working directory inside the container
WORKDIR /app

# Install system dependencies: ffmpeg (for pydub), and sox (for the sox Python package)
# --no-install-recommends helps keep the image size down
# rm -rf /var/lib/apt/lists/* cleans up apt cache to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg sox g++ wget && \
    rm -rf /var/lib/apt/lists/*


# Copy the requirements file into the container
COPY requirements.txt .

# --- IMPORTANT CHANGES HERE FOR DEPENDENCY ORDERING ---
# 1. Install PyTorch CPU-only version first to avoid CUDA dependencies.
#    The `nemo_toolkit[asr]` dependency on torch will then use this already-installed CPU version.
RUN pip install --no-cache-dir torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu

# 2. Install numpy explicitly before other dependencies, as some packages (like sox)
#    might require numpy to be present during their installation/metadata generation.
RUN pip install --no-cache-dir numpy==1.26.4 Cython

# RUN apt-get install g++ -y 
# 3. Install the rest of the Python dependencies from requirements.txt.
#    'torch' and 'numpy' should now be removed from from requirements.txt as they're installed above.
RUN pip install --no-cache-dir -r requirements.txt

RUN pip uninstall huggingface-hub -y
RUN pip install huggingface-hub==0.20.0



RUN mkdir -p nemo_models

# --- Download Models using wget from Hugging Face Hub ---
ENV NEMO_MODEL_URL="https://huggingface.co/n-log-n/nemo-hindi-asr/resolve/main/stt_hi_conformer_ctc_medium.nemo"
ENV ONNX_MODEL_URL="https://huggingface.co/n-log-n/nemo-hindi-asr/resolve/main/stt_hi_conformer_ctc_medium.onnx"

# Download the .nemo model directly into the nemo_models directory
RUN wget -O nemo_models/stt_hi_conformer_ctc_medium.nemo ${NEMO_MODEL_URL} \
    && echo "Downloaded NeMo model."

# Download the .onnx model directly into the nemo_models directory
RUN wget -O nemo_models/stt_hi_conformer_ctc_medium.onnx ${ONNX_MODEL_URL} \
    && echo "Downloaded ONNX model."

# Optional: Verify download (useful for debugging Docker builds)
RUN echo "Contents of nemo_models/:" && ls -lh nemo_models/



# Copy the rest of your application files
# This includes main.py, your nemo_models/ directory, and the static/ directory
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container
# --port 8000 specifies the port
# We use the non-reload version for production-like environments
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# docker build -t hindi-asr-app .
# docker run -p 8000:8000 hindi-asr-app