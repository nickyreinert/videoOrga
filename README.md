# Video Auto-Tagger

Automatically analyze and tag your video collection using local AI models running on your GPU. All metadata is stored in a SQLite database with thumbnail previews, ready for building a web browser interface.

## Features

- 🎬 Extract representative frames from videos
- 🤖 Analyze frames using local AI (BLIP, BLIP-2, or CLIP)
- 🏷️ Automatically generate descriptive tags
- 🎤 **Audio transcription with Whisper** (optional) ⭐ NEW
- 📝 **AI text summarization** (optional) ⭐ NEW
- 💾 Store metadata in SQLite database
- 📸 Generate 3 thumbnail previews per video (base64 encoded)
- 📅 Smart datetime parsing from filenames
- 📊 Rich file metadata (size, dates, resolution, codec)
- 🔍 Search videos by tags and transcripts
- ⚡ GPU-accelerated (optimized for RTX 3070)
- 📦 Batch processing for entire directories
- 📈 Database statistics and analytics

## Requirements

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (tested on RTX 3070)
- 8GB+ GPU VRAM recommended
- FFmpeg (for video processing)

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 2. Install PyTorch with CUDA

Visit [PyTorch website](https://pytorch.org/get-started/locally/) or use:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify GPU Support

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Process a Single Video

```bash
python video_tagger.py my_video.mp4
```

This creates a `video_archive.db` SQLite database in the same directory as your video.

### Process All Videos in a Directory

```bash
python video_tagger.py /path/to/videos/ --recursive
```

This creates `video_archive.db` inside the `/path/to/videos/` directory.

### Search Videos by Tag

```bash
python video_tagger.py . --search "outdoor"
```

### View Statistics

```bash
python video_tagger.py . --stats
```

## Usage

```
python video_tagger.py <input> [options]

Arguments:
  input                 Video file or directory to process (default: current dir)

Options:
  --frames N            Number of frames to extract (default: 8)
  --model MODEL         AI model: blip, blip2, or clip (default: blip)
  --db PATH             SQLite database path (default: video_archive.db)
  --recursive           Process subdirectories
  --force               Reprocess already tagged videos
  --search TAG          Search videos by tag
  --stats               Show database statistics
```

## Examples

### Basic Processing

```bash
# Process with default settings (BLIP model, 8 frames)
python video_tagger.py vacation.mp4

# Use more frames for longer videos
python video_tagger.py documentary.mp4 --frames 16

# Use BLIP-2 for better descriptions (needs more VRAM)
python video_tagger.py video.mp4 --model blip2
```

### Batch Processing

```bash
# Process all videos in current directory
python video_tagger.py .

# Process directory and subdirectories
python video_tagger.py /videos --recursive

# Use custom database location
python video_tagger.py /videos --db /my/custom/archive.db

# View what's in the database
python video_tagger.py . --stats
```

### Searching

```bash
# Find all outdoor videos
python video_tagger.py . --search "outdoor"

# Find videos with food
python video_tagger.py . --search "food"

# Find videos with people
python video_tagger.py . --search "people"
```

## AI Models

### BLIP (Default - Recommended)
- **Speed**: Fast ⚡
- **VRAM**: ~3-4GB
- **Quality**: Good descriptions
- **Best for**: Most use cases

### BLIP-2
- **Speed**: Slower
- **VRAM**: ~6-7GB
- **Quality**: More detailed descriptions
- **Best for**: When you need detailed analysis

### CLIP
- **Speed**: Fastest ⚡⚡
- **VRAM**: ~2-3GB
- **Quality**: Category classification
- **Best for**: Simple categorization, large batches

## Database Schema

The SQLite database contains the following tables:

### videos
- Complete file metadata (path, size, dates, hash)
- **Smart datetime parsing** from filenames (e.g., `VID_20231215_142530.mp4`)
- Video properties (duration, fps, resolution, codec)
- Processing information

### tags
- Video tags with confidence scores
- Indexed for fast searching

### thumbnails
- 3 random thumbnail frames per video
- Base64-encoded JPEG images
- Ready for web display

### frame_descriptions
- AI-generated descriptions for each analyzed frame

Example filename patterns recognized:
- `VID_20231215_142530.mp4` → 2023-12-15 14:25:30
- `2023-12-15_14-25-30.mp4` → 2023-12-15 14:25:30
- `20231215_142530.mp4` → 2023-12-15 14:25:30
- `video-2023-12-15.mp4` → 2023-12-15

## Tips for Best Results

1. **Frame Count**: Use 8-12 frames for short videos, 16+ for longer content
2. **Model Selection**: Start with BLIP, upgrade to BLIP-2 if needed
3. **GPU Memory**: Close other GPU applications before processing
4. **Video Quality**: Higher resolution videos produce better tags
5. **Batch Processing**: Process videos overnight for large collections

## Troubleshooting

### CUDA Out of Memory
- Use fewer frames: `--frames 6`
- Switch to smaller model: `--model blip` or `--model clip`
- Process one video at a time

### Slow Processing
- Use faster model: `--model clip`
- Reduce frame count: `--frames 6`
- Check GPU is being used: `nvidia-smi`

### Import Errors
```bash
# Reinstall transformers
pip install --upgrade transformers accelerate

# For BLIP-2
pip install salesforce-lavis
```

## Project Structure

```
video-auto-tagger/
├── frame_extractor.py    # Frame sampling & thumbnail generation
├── ai_analyzer.py        # AI model integration
├── db_handler.py         # SQLite database operations ⭐ NEW
├── video_tagger.py       # Main processing script
├── requirements.txt      # Dependencies
├── README.md            # This file
├── DEV-GUIDE.md         # Development roadmap
└── video_archive.db     # Generated SQLite database (output)
```

## Performance

Approximate processing times on RTX 3070:

- **Short video** (2-3 min): ~15-30 seconds
- **Medium video** (10 min): ~30-60 seconds
- **Long video** (30+ min): ~1-2 minutes

## Future Improvements

- [ ] **Flask Web UI for browsing** (NEXT: Browse tagged videos with thumbnails!)
- [ ] Multi-threaded batch processing
- [ ] Scene change detection
- [ ] Audio analysis integration
- [ ] Custom tag vocabularies
- [ ] Video player integration
- [ ] Tag editing interface
- [ ] Export to various formats

## License

MIT License - Feel free to use and modify!

## Contributing

Contributions welcome! See DEV-GUIDE.md for development roadmap.

## Support

For issues or questions, check:
- DEV-GUIDE.md for development details
- GitHub issues
- Transformers documentation