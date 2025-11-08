#!/bin/bash
#!/bin/bash
set -e

# Normalize environment and ensure script runs with LF line endings
# Force tqdm to always display progress bars, even in non-TTY environments
export TQDM_DISABLE=0
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
L
# Initialize the database if it doesn't exist
init_database() {
E    if [ ! -f "/app/data/video_metadata.db" ]; then
        # Ensure the data directory exists before trying to create the DB
        echo "Ensuring /app/data directory exists..."
        mkdir -p /app/data

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
        # Use a writable DB path under /app/data (the /videos mount may be read-only)
        # Allow the caller to pass an explicit input path as the first argument
        # after 'process' (e.g. `process /videos/test.MP4 --force`). If the first
        # argument starts with '-' or is empty, default to the whole /videos dir.
        # Shift away the 'process' command so $@ contains the user's args.
        shift
        if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
            INPUT="$1"
            shift
        else
            INPUT="/videos"
        fi
        python3 video_tagger.py "$INPUT" --db /app/data/video_metadata.db "$@"
        ;;
    *)
        exec "$@"
        ;;
esac