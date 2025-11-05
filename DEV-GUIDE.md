# Video Auto-Tagger Development Guide

## Project Overview
Automated video tagging system that extracts frames from videos, analyzes them using local AI (GPU-accelerated), and generates descriptive tags/metadata.

## System Requirements
- Python 3.8+
- NVIDIA GeForce RTX 3070 (CUDA-compatible)
- FFmpeg for video processing
- Local AI model for image analysis

## Technology Stack
- **Video Processing**: OpenCV / FFmpeg
- **AI Model**: BLIP-2, LLaVA, or similar vision-language model
- **GPU Acceleration**: PyTorch with CUDA
- **Storage**: JSON/SQLite for metadata
- **Dependencies**: transformers, opencv-python, torch, pillow

## Development Phases

### Phase 1: Environment Setup ✓
- [x] Create project structure
- [x] Document system requirements
- [x] List dependencies
- [ ] Create virtual environment setup script
- [ ] Install CUDA-enabled PyTorch
- [ ] Install video processing libraries

### Phase 2: Video Frame Extraction ✓
- [x] Implement frame sampling logic
- [x] Support multiple sampling strategies (uniform, intelligent)
- [ ] Test with various video formats
- [ ] Optimize frame extraction performance
- [ ] Add progress tracking

### Phase 3: Local AI Integration
- [ ] Research and select optimal model for RTX 3070
  - BLIP-2 (good balance of speed/quality)
  - LLaVA (more detailed descriptions)
  - CLIP (fast, tag-focused)
- [ ] Implement model loading with GPU support
- [ ] Create frame analysis pipeline
- [ ] Batch processing for efficiency
- [ ] Memory management for GPU

### Phase 4: Tag Generation & Storage ✓
- [x] Define tag extraction logic
- [x] Create metadata storage structure (JSON)
- [x] Implement SQLite storage with rich metadata
- [x] Extract file metadata (size, dates, etc.)
- [x] Smart datetime parsing from filenames
- [x] Generate and store thumbnail previews
- [ ] Tag deduplication and normalization
- [ ] Confidence scoring for tags

### Phase 5: Batch Processing & CLI
- [ ] Create command-line interface
- [ ] Support directory scanning
- [ ] Progress bars and logging
- [ ] Error handling and recovery
- [ ] Resume capability for interrupted jobs

### Phase 6: Optimization & Features
- [ ] Multi-threading for video processing
- [ ] GPU batch optimization
- [ ] Cache processed videos (checksums)
- [ ] Export formats (CSV, JSON, database)
- [ ] Web UI (optional, future)

## File Structure
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

## Current Progress

### ✅ Completed
1. Project structure defined
2. Basic frame extraction logic implemented
3. JSON metadata storage format created
4. Development roadmap documented

### 🔄 In Progress
- Setting up AI model integration

### 📋 Next Steps
1. Create `requirements.txt` with all dependencies
2. Implement AI model loader (recommend starting with BLIP-2)
3. Test frame extraction with sample videos
4. Build end-to-end pipeline for single video
5. Add batch processing capabilities

## Installation Commands (To Be Tested)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install PyTorch with CUDA 11.8 (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install opencv-python pillow transformers accelerate
pip install salesforce-lavis  # For BLIP-2
```

## Key Design Decisions

### Frame Sampling Strategy
- **Uniform Sampling**: Extract N frames evenly distributed (implemented)
- **Smart Sampling**: Detect scene changes, avoid similar frames (future)
- Recommended: 5-10 frames per video for balance of speed/accuracy

### AI Model Selection Criteria
- Must run efficiently on RTX 3070 (8GB VRAM)
- Should generate descriptive tags, not full captions
- Prefer models with good zero-shot classification
- BLIP-2 recommended as starting point

### Metadata Format
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

## Performance Targets
- Process 1 video (2-3 min length) in < 30 seconds
- Use < 6GB GPU memory
- Handle 100+ videos in batch without intervention

## Testing Plan
1. Test with single short video (< 1 min)
2. Test with various formats (MP4, AVI, MOV)
3. Test with long videos (> 10 min)
4. Stress test GPU memory with large batches
5. Validate tag quality manually on sample set

## Known Limitations & Future Improvements
- Currently no audio analysis (could add speech-to-text)
- No temporal understanding (frame-by-frame only)
- Tag vocabulary not customizable yet
- No web interface for browsing tagged videos

## Notes for AI Coding Agents
- Always maintain GPU memory efficiency
- Add error handling for corrupted videos
- Log all processing steps for debugging
- Keep dependencies minimal and well-documented
- Prefer configuration files over hardcoded values