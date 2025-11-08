#!/bin/bash
set -e

# Configuration
VENV=/cache/venv
PY=$VENV/bin/python
PIP=$VENV/bin/pip
REQUIREMENTS=/tmp/requirements.txt
WHEEL_DIR=/cache/torch
HASH_FILE=/app/data/.requirements_hash
WHEEL_HASH_FILE=/app/data/.wheel_hash

mkdir -p /cache/venv /cache/torch /app/data /cache/pip

# Create venv if missing
[ ! -d "$VENV" ] && python3 -m venv "$VENV"

# Ensure pip in venv is up to date
$PIP install --upgrade pip setuptools wheel

# Check if requirements changed
REQ_HASH=$(sha256sum "$REQUIREMENTS" | awk '{print $1}')
STORED_REQ_HASH=""
[ -f "$HASH_FILE" ] && STORED_REQ_HASH=$(cat "$HASH_FILE")

if [ "$REQ_HASH" != "$STORED_REQ_HASH" ]; then
    echo "Installing requirements..."
    $PIP install --no-cache-dir -r "$REQUIREMENTS"
    echo "$REQ_HASH" > "$HASH_FILE"
else
    echo "Requirements unchanged. Skipping install."
fi

# Check wheel cache
WHEEL_HASH=$(find "$WHEEL_DIR" -type f -exec sha256sum {} + | sha256sum | awk '{print $1}')
STORED_WHEEL_HASH=""
[ -f "$WHEEL_HASH_FILE" ] && STORED_WHEEL_HASH=$(cat "$WHEEL_HASH_FILE")

if [ "$WHEEL_HASH" != "$STORED_WHEEL_HASH" ]; then
    echo "Installing wheel packages..."
    for whl in "$WHEEL_DIR"/*.whl; do
        [[ $whl == *numpy* ]] && continue
        $PIP install --no-index "$whl" --no-deps
    done
    echo "$WHEEL_HASH" > "$WHEEL_HASH_FILE"
else
    echo "Wheel packages unchanged. Skipping install."
fi

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
    echo "Launching serve..."
    $PY -c "import torch, numpy; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'NumPy:', numpy.__version__)"
    wait_for_gpu
    init_database
    $PY web/app.py
    ;;
process)
    wait_for_gpu
    init_database
    INPUT=${2:-/videos}
    shift
    $PY video_tagger.py "$INPUT" --db /app/data/video_metadata.db "$@"
    ;;
*)
    exec "$@"
    ;;
esac
