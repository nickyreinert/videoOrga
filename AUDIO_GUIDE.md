# Audio Analysis Guide

## Overview

The video tagger now supports **automatic speech transcription** and **text summarization**! This allows you to:

- 🎤 Transcribe spoken words from videos
- 📝 Generate summaries of long transcripts
- 🔍 Search videos by spoken content
- 🏷️ Generate tags from speech (coming soon)
- 🌍 Auto-detect language

## Installation

### 1. Install FFmpeg (Required)

FFmpeg is needed to extract audio from videos.

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/ and add to PATH

**Verify installation:**
```bash
ffmpeg -version
```

### 2. Install Whisper

```bash
pip install openai-whisper
```

### 3. Optional: Install Summarization Model

The summarization model will be downloaded automatically on first use (~1GB).

## Usage

### Basic Usage

Add the `--audio` flag to enable transcription:

```bash
# Single video with audio analysis
python video_tagger.py my_video.mp4 --audio

# Batch processing with audio
python video_tagger.py /videos --recursive --audio
```

### Choose Whisper Model Size

Different Whisper models trade speed for accuracy:

| Model    | VRAM  | Speed      | Quality | Command |
|----------|-------|------------|---------|---------|
| tiny     | ~1GB  | Fastest ⚡⚡⚡ | Basic   | `--whisper-model tiny` |
| **base** | ~1GB  | Fast ⚡⚡   | Good ⭐  | (default) |
| small    | ~2GB  | Medium ⚡  | Better  | `--whisper-model small` |
| medium   | ~5GB  | Slow      | Great   | `--whisper-model medium` |
| large    | ~10GB | Slowest   | Best    | `--whisper-model large` |

**Recommended for RTX 3070: base or small**

```bash
# Use tiny model for speed
python video_tagger.py video.mp4 --audio --whisper-model tiny

# Use small model for better quality
python video_tagger.py video.mp4 --audio --whisper-model small
```

## What Gets Stored

When audio analysis is enabled, the database stores:

1. **Full Transcript** - Raw text of all spoken words
2. **Summary** - AI-generated summary (if transcript is long enough)
3. **Language** - Detected language (en, de, es, etc.)
4. **Word Count** - Number of words transcribed
5. **Has Speech Flag** - Boolean indicating if speech was detected

## Search by Spoken Content

Once videos are transcribed, you can search by what was said:

```bash
# Search for videos mentioning "vacation"
python video_tagger.py . --search "vacation"

# This searches in:
# - Filenames
# - Visual descriptions
# - Transcripts ⭐ NEW
# - Summaries ⭐ NEW
```

## Examples

### Example 1: Tutorial Videos

```bash
python video_tagger.py tutorials/ --recursive --audio --whisper-model small
```

**Result:**
- Transcribes instructor speech
- Summarizes key teaching points
- Searchable by topic mentioned

### Example 2: Family Videos

```bash
python video_tagger.py family_videos/ --recursive --audio
```

**Result:**
- Captures conversations
- Identifies who's speaking (future feature)
- Search by remembered phrases

### Example 3: Meeting Recordings

```bash
python video_tagger.py meetings/ --recursive --audio --whisper-model medium
```

**Result:**
- Full meeting transcripts
- AI-generated summaries
- Search by discussion topics

## Database Schema Updates

### Videos Table - New Columns

```sql
has_speech INTEGER          -- 0 or 1
transcript TEXT             -- Full transcript
transcript_summary TEXT     -- AI summary
audio_language TEXT         -- Detected language (en, de, etc.)
word_count INTEGER          -- Words transcribed
```

### Search Queries

The `search_videos()` function now includes:
- Transcript text matching
- Summary text matching
- Filter by `has_speech`

## Performance

### Processing Time Examples

**2-minute video with speech (base model):**
- Audio extraction: 2-3 seconds
- Transcription: 10-15 seconds
- Summarization: 3-5 seconds
- **Total: ~20 seconds**

**10-minute video (base model):**
- Total: ~60-90 seconds

### VRAM Usage

- **Video analysis only**: ~3-4GB
- **+ Audio (base)**: ~4-5GB
- **+ Audio (small)**: ~5-6GB

You can run both on RTX 3070!

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

### 4. Videos Without Speech

If no speech is detected:
- `has_speech = 0`
- Transcript and summary remain empty
- No performance impact on later searches

## Troubleshooting

### FFmpeg Not Found

**Error:** `FFmpeg not found`

**Solution:**
```bash
# Install FFmpeg (see Installation section)
# Verify it's in PATH:
ffmpeg -version
```

### Out of Memory (CUDA)

**Error:** `CUDA out of memory`

**Solutions:**
1. Use smaller Whisper model: `--whisper-model tiny`
2. Process videos one at a time (not batch)
3. Close other GPU applications
4. Use CPU mode (slower):
   - Set `device="cpu"` in `audio_analyzer.py`

### Slow Processing

**Problem:** Transcription takes too long

**Solutions:**
1. Use faster model: `--whisper-model tiny`
2. Ensure GPU is being used (check `nvidia-smi`)
3. Process overnight for large collections

### Poor Transcription Quality

**Problem:** Transcript has many errors

**Solutions:**
1. Use better model: `--whisper-model small` or `medium`
2. Ensure audio quality is good (check original video)
3. Specify language if auto-detection fails

## Advanced Usage

### Reprocess Audio Only

To re-transcribe videos with better model:

```bash
python video_tagger.py . --force --audio --whisper-model medium
```

### Export Transcripts

Currently stored in database. To export:

```python
from db_handler import DatabaseHandler

db = DatabaseHandler("video_archive.db")
videos = db.search_videos(has_speech=True)

for video in videos:
    print(f"=== {video['file_name']} ===")
    print(video['transcript'])
    print()
```

### Filter by Speech

Search only videos with speech:

```python
# In Python
db.search_videos(has_speech=True, search_text="vacation")
```

## What's Next?

Planned audio features:

- [ ] Speaker diarization (identify who's speaking)
- [ ] Extract tags from transcript
- [ ] Sentiment analysis
- [ ] Topic extraction
- [ ] Timestamp-based search
- [ ] Export SRT subtitle files
- [ ] Multi-language support in web UI

## Statistics

After processing with `--audio`, view stats:

```bash
python video_tagger.py . --stats
```

**Output includes:**
```
Video Archive Statistics
========================================
Total videos: 127
Videos with speech: 89
Total words transcribed: 45,382
Unique tags: 342
...
```

## Example Workflow

### Complete Video Archive Pipeline

```bash
# 1. Initial processing (fast)
python video_tagger.py ~/Videos --recursive --audio

# 2. View results
python video_tagger.py . --stats

# 3. Search by content
python video_tagger.py . --search "birthday party"

# 4. Find all videos with speech
# (will be added to web UI)
```

## Flask Web UI Integration

The audio data is **ready for web display**:

- Show transcript on video detail page
- Display summary as description
- Search bar includes transcript search
- Filter by language
- "Has speech" badge on thumbnails

All database queries already support audio data!

## Technical Details

### Whisper Models

Whisper is OpenAI's open-source speech recognition model:
- Trained on 680,000 hours of multilingual data
- Supports 99 languages
- Runs locally on your GPU
- No API calls or internet needed

### Summarization Model

Uses DistilBART (distilled BART):
- Smaller, faster version of Facebook's BART
- ~1GB model size
- Runs on GPU
- Extractive + abstractive summarization

### Audio Format

Extracted audio is converted to:
- Format: WAV (PCM 16-bit)
- Sample rate: 16kHz (Whisper's preferred rate)
- Channels: Mono
- Temporary files are cleaned up automatically

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