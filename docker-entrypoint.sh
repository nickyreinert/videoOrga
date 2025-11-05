#!/bin/bash
set -e

# Normalize environment and ensure script runs with LF line endings
export LC_ALL=C.UTF-8

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
    if [ ! -f "/app/data/video_metadata.db" ]; then
        echo "Initializing database..."
        python3 -c "from db_handler import init_db; init_db()"
    fi
}

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
        python3 video_tagger.py "/videos" --recursive ${@:2}
        ;;
    *)
        exec "$@"
        ;;
esac