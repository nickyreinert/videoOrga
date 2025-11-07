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
docker compose run --rm video-tagger process "/videos/test.MP4" --audio --force 

docker compose run --rm video-tagger process "/videos/1997/"  --audio --force --recursive

docker compose run --rm video-tagger process "/videos/1997/"  --audio --force --language=de 

# Check GPU status
docker exec video-tagger nvidia-smi

# If needed, restart the NVIDIA container runtime
sudo systemctl restart nvidia-dockerdocker-compose run --rm video-tagger rm -rf /app/data/cache/*

# Backup database
docker cp video-tagger:/app/data/video_metadata.db ./backup/