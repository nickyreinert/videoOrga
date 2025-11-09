#!/bin/bash
set -e

echo "Starting entrypoint script..."

# Function to handle errors during setup
installation_error_handler() {
    echo "-------------------------------------------------"
    echo "!!! An error occurred during dependency installation."
    echo "!!! Cleaning up the virtual environment to force a fresh install on next run."
    rm -rf "/cache/venv"
    echo "!!! Virtual environment at /cache/venv has been removed."
    echo "!!! Please try running the command again."
    echo "-------------------------------------------------"
    exit 1
}

# Set the trap to catch errors only during the installation phase
trap installation_error_handler ERR

# Configuration
VENV=/cache/venv
PY=$VENV/bin/python
PIP=$VENV/bin/pip
REQUIREMENTS=/tmp/requirements.txt
WHEEL_DIR=/cache/torch
HASH_FILE=/app/data/.requirements_hash
WHEEL_HASH_FILE=/app/data/.wheel_hash

mkdir -p /cache/venv /cache/torch /app/data /cache/pip

echo "Using virtual environment at $VENV"
# Create venv if it's missing or incomplete (e.g., pip is not found)
if [ ! -f "$PIP" ]; then
    echo "Virtual environment is missing or incomplete. Creating a new one..."
    rm -rf "$VENV"
    python3 -m venv "$VENV"
fi

# Check if requirements or constraints have changed
REQ_HASH=$(sha256sum "$REQUIREMENTS" /app/constraints.txt | sha256sum | awk '{print $1}')
STORED_REQ_HASH=""
[ -f "$HASH_FILE" ] && STORED_REQ_HASH=$(cat "$HASH_FILE")

echo "Current requirements hash: $REQ_HASH"
if [ "$REQ_HASH" != "$STORED_REQ_HASH" ]; then
    echo "Requirements have changed. Installing dependencies..."
    
    # Install all dependencies in a single, atomic command.
    # This allows pip to resolve the entire dependency tree correctly.
    # It installs torch from the special URL, packages from requirements.txt,
    # and enforces the numpy version from constraints.txt.
    $PIP install --cache-dir /cache/pip \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
        -r "$REQUIREMENTS" \
        -c /app/constraints.txt

    echo "Installation complete."
    echo "$REQ_HASH" > "$HASH_FILE"
else
    echo "Requirements unchanged. Skipping install."
fi

# Disable the trap now that installation is done
trap - ERR
echo "Dependency setup finished."


# Function to wait for GPU
wait_for_gpu() {
    echo "Checking GPU availability..."
    until command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; do
        echo "Waiting for GPU..."
        sleep 2
    done
    echo "GPU available"
}

# Initialize database if missing
init_database() {
    DB=/app/data/video_metadata.db
    if [ ! -f "$DB" ]; then
        mkdir -p /app/data
        $PY -c "from db_handler import init_db; init_db()"
    fi
}

case "$1" in
serve)
    echo "Launching server..."
    $PY -c "import torch, numpy; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'NumPy:', numpy.__version__)"
    wait_for_gpu
    init_database
    $PY web/app.py
    ;;
process)
    echo "Launching video processing..."
    echo "Waiting for GPU..."
    wait_for_gpu
    echo "Initializing database..."
    init_database
    echo "Processing videos..."
    shift # Remove 'process' from the arguments
    # The first argument is now the video file/directory
    $PY video_tagger.py "$@" --db /data/video_metadata.db
    ;;
*)
    exec "$@"
    ;;
esac
