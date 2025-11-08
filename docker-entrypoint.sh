#!/bin/bash
set -e

# --- Configuration ---
# These environment variables are set in the Dockerfile for consistency.
# WHEELS_DIR: Directory inside the image with pre-downloaded .whl files.
# PYTHON_PACKAGES_DIR: Persistent volume mount path for installed packages.
REQUIREMENTS_HASH_FILE="$PYTHON_PACKAGES_DIR/.requirements_hash"

echo "--- Docker Entrypoint: Package Installation Check ---"

# Ensure the persistent package directory exists
# This directory is mounted from the host via docker-compose.
mkdir -p "$PYTHON_PACKAGES_DIR"

# --- Smart Installation Logic ---
# Calculate the hash of the current requirements.txt from the image.
current_hash=$(sha256sum /tmp/requirements.txt | awk '{ print $1 }')
# Check for a stored hash in the persistent package directory.
stored_hash=$(cat "$REQUIREMENTS_HASH_FILE" 2>/dev/null || echo "")

if [ "$current_hash" != "$stored_hash" ]; then
    echo "Requirements have changed (or this is the first run). Installing packages..."
    echo "This will only happen when requirements.txt is modified."

    # Install packages from the pre-downloaded wheels into the persistent volume.
    # --no-index: Prevents pip from accessing the network.
    # --find-links: Specifies the local directory containing the wheels.
    # --target: Installs packages into the specified persistent directory.
    pip install \
        --no-index \
        --find-links="$WHEELS_DIR" \
        --target="$PYTHON_PACKAGES_DIR" \
        -r /tmp/requirements.txt

    echo "Package installation complete."
    # Store the new hash to mark this version of requirements as installed.
    echo "$current_hash" > "$REQUIREMENTS_HASH_FILE"
    echo "Updated requirements hash."
else
    echo "Packages are up-to-date. Skipping installation."
fi

# --- Python Path Configuration ---
# Add the persistent package directory to Python's path for this session.
export PYTHONPATH="$PYTHON_PACKAGES_DIR${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH set to: $PYTHONPATH"
echo "----------------------------------------------------"

# Function to wait for GPU availability
wait_for_gpu() {
    echo "Checking GPU availability..."
    while ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; do
        echo "Waiting for GPU to become available..."
        sleep 2
    done
    echo "GPU is available"
}

# Initialize the database if it doesn't exist
init_database() {
    # Use the VIDEO_DB_PATH env var, with a fallback.
    DB_PATH=${VIDEO_DB_PATH:-/data/video_metadata.db}
    if [ ! -f "$DB_PATH" ]; then
        echo "Initializing database..."
        python3 -c "from db_handler import init_db; init_db()"
    fi
}
# --- Command Execution ---
case "$1" in
    serve)
        wait_for_gpu
        init_database
        echo "Starting Flask server..."
        python3 web/app.py
        ;;
    process)
        wait_for_gpu
        init_database
        echo "Processing videos..."        
        # Shift away the 'process' command so $@ contains the user's args.
        shift
        if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
            INPUT="$1"
            shift
        else
            INPUT="/videos"
        fi
        python3 video_tagger.py "$INPUT" --db "${VIDEO_DB_PATH:-/data/video_metadata.db}" "$@"
        ;;
    *)
        exec "$@"
        ;;
esac