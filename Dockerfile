# Development Dockerfile
# Only installs dependencies, source code is bind-mounted
# CRITICAL: DO NOT CHANGE VERSION HERE, CUDA VERSION DEEPENDS ON TORCH, TORCHVISION and NUMPY VERSIONS BELOW
# REFER to https://pytorch.org/get-started/previous-versions/ to see compatible versions
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3-dev \
    ffmpeg \
    git \
    dos2unix \
    wget \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Set environment variables   
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    HF_HOME=/root/.cache/huggingface \
    WHISPER_CACHE=/root/.cache/whisper

WORKDIR /app

# Set the entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command for development
CMD ["/bin/bash"]