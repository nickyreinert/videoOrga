# Video Auto-Tagger

Automatically analyze and tag your video collection using local AI models running on your GPU. All metadata is stored in a SQLite database with thumbnail previews and audio transcription. Comes with a web frontend to browse the database.

## Features

- 🎬 Extract representative frames from videos
- 🤖 Analyze frames using local AI (Multimodal LLM)
- 🏷️ Automatically generate descriptive tags
- 🎤 Audio transcription with Whisper (optional)
- 📝 AI text summarization (optional)
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
- FFmpeg (for audio features)

...or **Docker**

## Installation

Refer do `DOCKER.md` to learn about Docker setup. Follow the steps below to install it directly on your machine. 

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

You may need to install an older Python version to run with the current **CUDA Version**. Right now this is **Python 3.11.8** You can use **PyEnv** for that:

First install some dependencies:

```
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev libncurses5-dev libncursesw5-dev \
libffi-dev liblzma-dev tk-dev
``` 

Then install **PyEnv**:
```
curl https://pyenv.run  -o install.sh
bash install.sh
```

Now install and activate Python 3.11.8:

```
pyenv install 3.11.8
pyenv global 3.11.8
```

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify GPU Support

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Install Audio-Support

For speech transcription:

```bash
# Install FFmpeg first
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/
```
### Prepare local cache for PyTorch

```bash
mkdir -p torch-cache
```

## Quick Start

### Process a Single Video

```bash
python video_tagger.py my_video.mp4
```

**What happens:**
1. Extracts 8 frames from the video
2. Analyzes them with AI (Multimodal LLM)
3. Generates descriptive tags
4. Extracts 3 thumbnail previews
5. Parses datetime from filename (if available)
6. Stores everything in `video_archive.db`

When audio analysis is enabled, the database stores:

1. **Full Transcript** - Raw text of all spoken words
2. **Summary** - AI-generated summary (if transcript is long enough)
3. **Language** - Detected language (en, de, es, etc.)
4. **Word Count** - Number of words transcribed
5. **Has Speech Flag** - Boolean indicating if speech was detected

### Process All Videos in a Directory

```bash
python video_tagger.py /path/to/videos/ --recursive
```

This creates `video_archive.db` inside the `/path/to/videos/` directory.

### Search Videos by Tag or spoken text

```bash
python video_tagger.py . --search "outdoor"
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

### Video Analysis (Multimodal LLM)

The system now uses a **single multimodal LLM** for all tasks: frame analysis, captioning, summarization, and tagging. This provides higher quality and more consistent results compared to using separate models.

#### LLaVA 1.5 (Default - Recommended)
- **Speed**: Fast ⚡
- **VRAM**: ~4-6GB (4-bit quantized)
- **Quality**: Excellent descriptions and summaries
- **Best for**: General purpose video tagging
- **Model**: `llava-hf/llava-1.5-7b-hf`

#### LLaVA 1.5 Large
- **Speed**: Slower
- **VRAM**: ~8-10GB (4-bit quantized)
- **Quality**: Superior detail and reasoning
- **Best for**: Complex scenes requiring detailed analysis
- **Model**: `llava-hf/llava-1.5-13b-hf`

#### BLIP-2 / InstructBLIP
- **Speed**: Medium
- **VRAM**: ~6-8GB
- **Quality**: Good alternative if LLaVA doesn't suit your needs
- **Best for**: Specific captioning styles

**Note:** The system automatically attempts to load models with **4-bit quantization** to save VRAM. If that fails, it falls back to 8-bit or full precision.

### Audio Model Comparison

Different Whisper models trade speed for accuracy:

| Model    | VRAM  | Speed      | Quality | Command |
|----------|-------|------------|---------|---------|
| tiny     | ~1GB  | Fastest ⚡⚡⚡ | Basic   | `--whisper-model tiny` |
| **base** | ~1GB  | Fast ⚡⚡   | Good ⭐  | (default) |
| small    | ~2GB  | Medium ⚡  | Better  | `--whisper-model small` |
| medium   | ~5GB  | Slow      | Great   | `--whisper-model medium` |
| large    | ~10GB | Slowest   | Best    | `--whisper-model large` |

**Recommended for RTX 3070: base or small**

Whisper is OpenAI's open-source speech recognition model:
- Trained on 680,000 hours of multilingual data
- Supports 99 languages
- Runs locally on your GPU
- No API calls or internet needed

### Customizing the AI Prompt

You can customize how the AI summarizes and tags videos by editing the `summary_prompt.txt` file. This allows you to:
- Change the target language instructions
- Adjust the style of the summary
- Focus on specific details

The prompt file supports placeholders like `{language}`, `{visual_context}`, and `{audio_transcript}`.

## Database Schema

The SQLite database contains the following tables:

### videos
- Complete file metadata (path, size, dates, hash)
- **Smart datetime parsing** from filenames (e.g., `VID_20231215_142530.mp4`)
- Video properties (duration, fps, resolution, codec)
- Processing information

```sql
has_speech INTEGER          -- 0 or 1
transcript TEXT             -- Full transcript
transcript_summary TEXT     -- AI summary
audio_language TEXT         -- Detected language (en, de, etc.)
word_count INTEGER          -- Words transcribed
```

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
2. **Model Selection**: Start with LLaVA (default), upgrade to Large if needed
3. **GPU Memory**: Close other GPU applications before processing
4. **Video Quality**: Higher resolution videos produce better tags
5. **Batch Processing**: Process videos overnight for large collections

## Troubleshooting

### FFmpeg Not Found

**Error:** `FFmpeg not found`

**Solution:**
```bash
# Install FFmpeg (see Installation section)
# Verify it's in PATH:
ffmpeg -version
```

### Poor Transcription Quality

**Problem:** Transcript has many errors

**Solutions:**
1. Use better model: `--whisper-model small` or `medium`
2. Ensure audio quality is good (check original video)
3. Specify language if auto-detection fails

### CUDA Out of Memory
- Use fewer frames: `--frames 6`
- Switch to smaller model: `--model blip` or `--model clip`
- Process one video at a time

### Slow Processing
- Use faster model: `--model clip`
- Reduce frame count: `--frames 6`
- Check GPU is being used: `nvidia-smi`

### Import Errors
Approximate processing times on RTX 3070:

- **Short video** (2-3 min): ~15-30 seconds
- **Medium video** (10 min): ~30-60 seconds
- **Long video** (30+ min): ~1-2 minutes

**2-minute video with speech (base model):**
- Audio extraction: 2-3 seconds
- Transcription: 10-15 seconds
- Summarization: 3-5 seconds
- **Total: ~20 seconds**

**10-minute video (base model):**
- Total: ~60-90 seconds

### VRAM Usage

- **Video analysis only**: ~4-6GB
- **+ Audio (base)**: ~5-7GB
- **+ Audio (small)**: ~6-8GB

## Tips & Best Practices

### 1. Choose Model Based on Content

- **Casual content** (vlogs, family): `base` or `tiny`
- **Clear speech** (tutorials, presentations): `base`
- **Accents or noise**: `small` or `medium`
- **Professional transcription**: `medium` or `large`

### 2. Language Detection

Whisper auto-detects language, but you can specify:

```python
# In audio_analyzer.py, modify initialization:
AudioAnalyzer(whisper_model="base", language="en")
```

### 3. Processing Strategy

**Fast initial pass:**
```bash
python video_tagger.py /videos --recursive --whisper-model tiny
```

**High-quality reprocess:**
```bash
python video_tagger.py /videos --recursive --audio --whisper-model medium --force
```


## Future Improvements

- [ ] Flask Web UI for browsing (NEXT: Browse tagged videos with thumbnails!)
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

## FAQ

**Q: Does this work offline?**  
A: Yes! Everything runs locally. Models download once, then work offline.

**Q: What languages are supported?**  
A: 99 languages including English, German, Spanish, French, Chinese, Japanese, etc.

**Q: Can I disable summarization?**  
A: Yes, set `summarize=False` in `analyze_video_audio()` call.

**Q: How accurate is the transcription?**  
A: Very good for clear speech. Accuracy depends on:
- Audio quality
- Background noise
- Accents
- Model size (larger = more accurate)

**Q: Does it slow down video processing?**  
A: Yes, adds ~10-30 seconds per minute of video (with base model).

**Q: Can I use it without a GPU?**  
A: Yes, but much slower. Set `device="cpu"` in code.

**Q: Is my RTX 3070 enough?**  
A: Yes! Use `base` or `small` model. You have 8GB VRAM, which is plenty.

## Known Quirks

### Prompt management

- right now the prompt is tuned to work with `LLaVA`, means it's using `USER` and `ASSISTANT` keywords, when switchting to like `Mistral` or `LLaMA` the requires `INST` instead
- the logic looks for `Tags:` to split the output of the summarizer, this is not fail safe
- the logic is somehow hardcoded to German language, this is not fail safe