# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# --- Environment Configuration ---
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_CACHE_DIR=/cache/pip \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    # Directory inside the image with pre-downloaded .whl files
    WHEELS_DIR=/opt/wheels \
    # Persistent volume mount path for installed packages
    PYTHON_PACKAGES_DIR=/opt/packages

# --- System Dependencies ---
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev ffmpeg git dos2unix wget \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN pip install --upgrade pip

RUN if [ ! -f /cache/torch/torch-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl ]; then \
     pip download torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 -d /cache/torch/; \
    fi

# no-clean prevents pip from deleti ng build dependencies, speeding up subsequent installs    
# RUN pip install --no-index --find-links=/cache --no-clean torch torchvision

RUN pip install torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cu118


# --- Build-Time Package Downloading ---
# Copy requirements and download all packages as wheels into the WHEELS_DIR.
# This layer is only re-built if requirements.txt changes.
COPY requirements.txt /tmp/requirements.txt
RUN pip download --dest "$WHEELS_DIR" -r /tmp/requirements.txt

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN dos2unix /usr/local/bin/docker-entrypoint.sh && chmod +x /usr/local/bin/docker-entrypoint.sh

# --- Application Setup ---
WORKDIR /app
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["/bin/bash"]
