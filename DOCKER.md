# Adjust your mounting point in docker-compose.yml:
```
    volumes:
      # Mount video directory (read-only)
      - type: bind
        source: /e/OneDrive/Projekte/Video
        target: /videos
        read_only: true
```

# Build the image
docker compose build

# Start the web service
docker compose up -d

# Watch the logs

docker compose logs -f --tail=200

# Process all videos in the directory
docker compose run --rm video-tagger process --audio

# Process a specific video with force flag
docker compose run --rm video-tagger process "/videos/test.MP4" --config config.json

docker compose run --rm video-tagger process "/videos/test.MP4" --audio --force 

docker compose run --rm video-tagger process "/videos/1997/"  --audio --force --recursive

docker compose run --rm video-tagger process "/videos/1997/"  --audio --force --language=de 

# Check GPU status
docker exec video-tagger nvidia-smi

# If needed, restart the NVIDIA container runtime
sudo systemctl restart nvidia-dockerdocker-compose run --rm video-tagger rm -rf /app/data/cache/*

# Backup database
docker cp video-tagger:/app/data/video_metadata.db ./backup/

## CLI Arguments
The Docker container forwards all command‑line arguments to the `video_tagger.py` entrypoint. The full set of supported arguments is:

```
--frames-per-minute N    Number of frames to extract per minute (default: 2.0)
--min-frames N          Minimum frames to extract (default: 3)
--max-frames N          Maximum frames to extract (default: 50)
--model MODEL           AI model to use (llava, llava-large, blip2, instructblip) (default: llava)
--language LANG         Tag language (default: en)
--db PATH               SQLite database path
--audio                 Enable audio transcription
--whisper-model MODEL   Whisper model size (tiny, base, small, medium, large) (default: base)
--audio-language LANG   Force audio transcription language
--recursive             Process subdirectories recursively
--force                 Force reprocessing of videos
--search TAG            Search videos by tag
--stats                 Show database statistics
--fix-tags              Reprocess tags for all videos (no re‑analysis)
--config FILE           Path to JSON configuration file
```

Example Docker command using several arguments:
```
 docker compose run --rm video-tagger process "/videos/example.mp4" \
    --audio --whisper-model small --language de --force
```