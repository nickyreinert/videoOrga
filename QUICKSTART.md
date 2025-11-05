# Quick Start Guide - Video Auto-Tagger

## 🚀 Setup (5 minutes)

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install PyTorch with CUDA (for RTX 3070)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### 2. (Optional) Install Audio Features

For speech transcription:

```bash
# Install FFmpeg first
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/

# Install Whisper
pip install openai-whisper
```

### 3. Verify GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Should output: `CUDA available: True`

## 📹 Process Your First Video

```bash
python video_tagger.py my_vacation_video.mp4
```

**What happens:**
1. Extracts 8 frames from the video
2. Analyzes them with AI (BLIP model)
3. Generates descriptive tags
4. Extracts 3 thumbnail previews
5. Parses datetime from filename (if available)
6. Stores everything in `video_archive.db`

**Output example:**
```
Processing: my_vacation_video.mp4
[1/5] Extracting file metadata...
  File size: 145.23 MB
  Parsed datetime from filename: 2023-08-15T14:30:00
[2/5] Extracting frames for analysis...
  Extracted 8 frames
[3/5] Extracting thumbnail previews...
  Extracted 3 thumbnails
[4/5] Analyzing frames with AI...
  Processing frame 8/8...
[5/5] Storing in database...

✓ Successfully processed!
  Video ID: 1
  Tags: beach, ocean, sunny, people, vacation
  Thumbnails: 3 stored
```

## 📁 Process Entire Directory

```bash
# Process all videos in a folder
python video_tagger.py /path/to/videos --recursive

# This creates video_archive.db inside /path/to/videos/
```

## 🔍 Search Your Collection

```bash
# Find all beach videos
python video_tagger.py . --search "beach"

# Output:
Found 5 video(s) with tag 'beach':

vacation_2023.mp4
  Path: /videos/vacation_2023.mp4
  Duration: 180.5s
  Date: 2023-08-15T14:30:00
  Tags: beach, ocean, sunny, people, vacation

summer_trip.mp4
  ...
```

## 📊 View Statistics

```bash
python video_tagger.py . --stats
```

**Output:**
```
Video Archive Statistics
========================================
Total videos: 127
Unique tags: 342
Total storage: 45.6 GB
Total duration: 8.3 hours

Top 20 Tags:
  outdoor: 45 videos
  indoor: 38 videos
  people: 67 videos
  food: 23 videos
  ...
```

## 🗃️ Database Structure

Your `video_archive.db` contains:

### Videos Table
- File metadata (path, size, dates)
- Parsed datetime from filename (smart!)
- Video properties (duration, resolution, codec)
- AI-generated description

### Tags Table
- All detected tags
- Searchable and indexed

### Thumbnails Table
- 3 preview images per video
- Base64-encoded (ready for web display)
- Frame numbers stored

### Frame Descriptions Table
- Detailed AI descriptions for each frame analyzed

## 🎯 Common Use Cases

### 1. Tag Family Videos
```bash
python video_tagger.py ~/Videos/Family --recursive
```

### 2. Find Specific Content
```bash
# Find all cooking videos
python video_tagger.py . --search "food"
python video_tagger.py . --search "kitchen"

# Find outdoor activities
python video_tagger.py . --search "outdoor"
```

### 3. Reprocess with Better Model
```bash
# Use BLIP-2 for more detailed analysis
python video_tagger.py . --model blip2 --force --recursive
```

### 4. Quick Analysis with CLIP
```bash
# Faster processing for large collections
python video_tagger.py . --model clip --frames 6 --recursive
```

## 💡 Tips

1. **First run is slower** - The AI model needs to download (~2-3GB)
2. **Start small** - Test with 1-2 videos first
3. **Use BLIP** - Best balance of speed and quality
4. **More frames = better tags** - But slower processing
5. **Filename matters** - Name files like `VID_20231215_142530.mp4` for auto-dating

## 🔧 Troubleshooting

### Out of GPU Memory
```bash
# Use fewer frames
python video_tagger.py video.mp4 --frames 6

# Or use faster model
python video_tagger.py video.mp4 --model clip
```

### Slow Processing
- Close other GPU applications
- Use `--model clip` for speed
- Reduce `--frames` count

### Can't Find Videos
- Check file extensions (mp4, avi, mov, mkv, etc.)
- Use `--recursive` for subdirectories

## 📈 What's Next?

**Flask Web Browser** - Coming next! Will allow you to:
- Browse videos with thumbnails in a grid
- Filter by tags, date, and search
- Play videos directly in browser
- Edit tags manually
- View statistics with charts

All the database infrastructure is ready for the web interface!

## 🎨 Filename Parsing Examples

The system automatically extracts dates from these patterns:

✅ `VID_20231215_142530.mp4` → 2023-12-15 14:25:30  
✅ `2023-12-15_14-25-30.mp4` → 2023-12-15 14:25:30  
✅ `20231215_142530.mp4` → 2023-12-15 14:25:30  
✅ `video-2023-12-15.mp4` → 2023-12-15  
✅ `IMG_20231215.mp4` → 2023-12-15  
❌ `my_vacation.mp4` → No date (uses file modification date instead)

## 📞 Need Help?

Check these files:
- `README.md` - Full documentation
- `DEV-GUIDE.md` - Development roadmap and technical details
- Database file: `video_archive.db` (can open with any SQLite browser)