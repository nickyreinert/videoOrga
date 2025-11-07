# Development Dockerfile
# Only installs dependencies, source code is bind-mounted

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    dos2unix \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support FIRST (this is the big download)
# This layer will be cached unless requirements change
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 numpy==1.26.4 --index-url https://download.pytorch.org/whl/cu118

# Create directories for caches and data
RUN mkdir -p /root/.cache/huggingface /root/.cache/whisper /data /videos

# Set environment variables   
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/root/.cache/huggingface \
    WHISPER_CACHE=/root/.cache/whisper

# --- COPY local files AFTER large installs ---

# Copy the entrypoint script and make it executable
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN dos2unix /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy requirements and install them
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

# Verify GPU is available
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || true

# Set the entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command for development
CMD ["/bin/bash"]