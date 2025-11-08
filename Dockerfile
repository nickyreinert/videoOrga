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
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy the entrypoint script and make it executable
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN dos2unix /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy requirements and install them
COPY requirements.txt /tmp/requirements.txt

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