# Video Auto-Tagger Development & Setup Guide

## Project Overview

Automated video tagging system that extracts frames from videos, analyzes them using local AI (GPU-accelerated), and generates descriptive tags/metadata.

## System Requirements

* Python 3.11.8 (via PyEnv)
* NVIDIA GeForce RTX 3070 (CUDA-compatible)
* FFmpeg for video processing
* Local AI model for image analysis

## Technology Stack

* **Video Processing**: OpenCV / FFmpeg
* **AI Model**: BLIP-2, LLaVA, or similar vision-language model
* **GPU Acceleration**: PyTorch with CUDA
* **Storage**: JSON/SQLite for metadata
* **Dependencies**: transformers, opencv-python, torch, pillow

## 2. File Structure

```
video-auto-tagger/
├── DEV-GUIDE.md              # This file
├── README.md                 # User documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Installation script
├── config.yaml               # Configuration file
├── src/
│   ├── __init__.py
│   ├── frame_extractor.py    # Video frame sampling ✓
│   ├── ai_analyzer.py        # AI model integration
│   ├── tag_manager.py        # Tag processing & storage ✓
│   └── batch_processor.py    # Main processing pipeline
├── models/                   # Local AI model cache
├── data/
│   └── video_metadata.json   # Output metadata ✓
└── tests/
    └── test_extraction.py
```

## 3. Running Video Tagging

Single video:

```bash
python video_tagger.py your_video.mp4
```

Entire directory:

```bash
python video_tagger.py /path/to/videos --recursive
```

## 4. Metadata Format

```json
{
  "video_path": "path/to/video.mp4",
  "processed_date": "2025-01-15T10:30:00",
  "duration_seconds": 120.5,
  "tags": ["outdoor", "table", "food", "daytime"],
  "confidence_scores": {"outdoor": 0.95, "table": 0.88},
  "frame_count_analyzed": 8
}
```

## 5. Performance Targets

* Process 1 video (2-3 min) in < 30 seconds
* Use < 6GB GPU memory
* Handle 100+ videos in batch

## 6. Notes for AI Coding Agents

* Maintain GPU memory efficiency
* Handle corrupted videos
* Log all steps for debugging
* Keep dependencies minimal
* Prefer config files over hardcoded values
