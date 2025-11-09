"""
Video Auto-Tagger - Main Script with SQLite Storage
Processes videos, extracts frames, analyzes with AI, and stores in SQLite database
NOW USING SINGLE MULTIMODAL LLM FOR EVERYTHING
"""

import os
import json
import argparse
from pathlib import Path
import warnings
from datetime import datetime
from typing import List, Dict

# Import our modules
from frame_extractor import FrameExtractor
from ai_analyzer import AIAnalyzer
from audio_analyzer import AudioAnalyzer
from db_handler import DatabaseHandler, extract_file_metadata


class VideoTagger:
    """Main class for processing and tagging videos with SQLite storage"""
    
    def __init__(self, 
                 num_frames: int = 8,
                 num_thumbnails: int = 5,
                 tag_language: str = 'en',
                 tag_stopwords: List[str] = None,
                 model_name: str = "llava",  # Changed from 'blip' to 'llava'
                 db_path: str = None,
                 enable_audio: bool = False,
                 whisper_model: str = "base",
                 language: str = None,
                 no_pre_detect: bool = False):
        """
        Initialize video tagger with SQLite backend
        
        Args:
            num_frames: Number of frames to extract per video
            num_thumbnails: Number of thumbnail previews to extract
            tag_language: Target language for tags (e.g., 'en', 'de', 'fr')
            tag_stopwords: Custom stopwords to filter
            model_name: Multimodal model ('llava', 'llava-large', 'blip2', 'instructblip')
            db_path: Path to SQLite database
            enable_audio: Whether to transcribe and summarize audio
            whisper_model: Whisper model size
            language: Language for transcription or None for auto-detect
            no_pre_detect: Disable language pre-detection
        """
        self.extractor = FrameExtractor(num_frames=num_frames, num_thumbnails=num_thumbnails)

        # Initialize with new multimodal analyzer (no need for separate LLM config)
        self.analyzer = AIAnalyzer(
            model_name=model_name,
            tag_language=tag_language,
            stopwords=tag_stopwords
        )
        
        # Audio analysis (optional)
        self.enable_audio = enable_audio
        if enable_audio:
            self.audio_analyzer = AudioAnalyzer(
                whisper_model=whisper_model,
                device="auto",
                language=language,
                no_pre_detect=no_pre_detect
            )
        else:
            self.audio_analyzer = None
        
        # Set database path
        if db_path is None:
            db_path = "video_archive.db"
        self.db = DatabaseHandler(db_path)
        
        print(f"Using database: {self.db.db_path}")
        if enable_audio:
            print(f"Audio transcription: ENABLED (Whisper {whisper_model})")
            if language:
                print(f"Language forced to: {language.upper()}")
    
    def process_video(self, video_path: str, force: bool = False) -> Dict:
        """
        Process a single video: extract frames, analyze, generate tags, store in DB
        
        Args:
            video_path: Path to video file
            force: Force reprocessing even if already in database
            
        Returns:
            Dictionary with video metadata and tags
        """
        video_path = str(Path(video_path).resolve())
        
        # Check if already processed
        if self.db.video_exists(video_path) and not force:
            print(f"Video already in database: {Path(video_path).name}")
            print("Use --force to reprocess")
            video_id = self.db.get_video_id(video_path)
            return self.db.get_video_with_tags(video_id)
        
        print(f"\n{'='*60}")
        print(f"Processing: {Path(video_path).name}")
        print(f"{'='*60}")
        
        # Step 1: Extract file metadata
        print("\n[1/5] Extracting file metadata...")
        try:
            file_meta = extract_file_metadata(video_path)
            print(f"  File size: {file_meta['file_size_bytes'] / (1024**2):.2f} MB")
            print(f"  Modified: {file_meta['file_modified_date']}")
            if file_meta['parsed_datetime']:
                print(f"  Parsed datetime from filename: {file_meta['parsed_datetime']}")
        except Exception as e:
            print(f"Error extracting file metadata: {e}")
            return None
        
        # Step 2: Extract frames for analysis
        print("\n[2/5] Extracting frames for analysis...")
        try:
            frames, video_meta = self.extractor.extract_frames(video_path)
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return None
        
        if not frames:
            print("No frames extracted, skipping video")
            return None
        
        # Step 3: Extract thumbnail previews
        print("\n[3/5] Extracting thumbnail previews...")
        try:
            thumbnails = self.extractor.extract_thumbnails(video_path)
        except Exception as e:
            print(f"Warning: Could not extract thumbnails: {e}")
            thumbnails = []
        
        # Step 4: Analyze frames with AI (multimodal LLM does everything)
        print("\n[4/5] Analyzing frames with multimodal AI...")
        try:
            analysis = self.analyzer.analyze_frames(frames)
        except Exception as e:
            print(f"Error analyzing frames: {e}")
            return None
        
        # Step 5: Analyze audio (optional)
        audio_result = {
            'has_speech': False,
            'transcript': '',
            'summary': '',
            'language': '',
            'word_count': 0
        }
        
        if self.enable_audio:
            print("\n[5/5] Analyzing audio (transcription)...")
            try:
                audio_result = self.audio_analyzer.analyze_video_audio(
                    video_path,
                    summarize=False  # We'll use our multimodal LLM for summary
                )
            except Exception as e:
                print(f"Warning: Audio analysis failed: {e}")
        else:
            print("\n[5/5] Skipping audio analysis (use --audio to enable)")
        
        # Generate consolidated AI summary using multimodal LLM
        print("\n[6/6] Generating consolidated AI summary...")
        ai_summary_result = self.analyzer.generate_ai_summary_and_tags(
            visual_descriptions=analysis['descriptions'],
            audio_transcript=audio_result.get('transcript', ''),
            language=self.analyzer.tag_language
        )
        
        ai_summary_text = ''
        if ai_summary_result:
            ai_summary_text = ai_summary_result['summary']
            # Merge AI-generated tags with frame tags
            analysis['tags'] = sorted(list(set(analysis['tags'] + ai_summary_result['tags'])))
            print(f"  Summary: {ai_summary_text[:100]}...")
            print(f"  Final tag count: {len(analysis['tags'])}")
        
        # Step 7: Store everything in database
        print("\n[7/7] Storing in database...")
        
        # Combine metadata
        video_data = {
            **file_meta,
            'duration_seconds': video_meta['duration_seconds'],
            'fps': video_meta['fps'],
            'width': video_meta['width'],
            'height': video_meta['height'],
            'resolution': video_meta['resolution'],
            'codec': video_meta.get('codec', 'unknown'),
            'processed_date': datetime.now().isoformat(),
            'frames_analyzed': analysis['frame_count'],
            'description': analysis['descriptions'][0] if analysis['descriptions'] else None,
            # Audio data
            'has_speech': audio_result['has_speech'],
            'transcript': audio_result['transcript'],
            'transcript_summary': audio_result['summary'],
            'ai_summary': ai_summary_text,
            'audio_language': audio_result['language'],
            'word_count': audio_result['word_count']
        }
        
        # Insert video record
        video_id = self.db.insert_video(video_data)
        
        # Insert tags
        self.db.insert_tags(video_id, analysis['tags'])
        
        # Insert frame descriptions
        self.db.insert_frame_descriptions(video_id, analysis['descriptions'])
        
        # Insert thumbnails
        if thumbnails:
            self.db.insert_thumbnails(video_id, thumbnails)
        
        # Retrieve complete record
        result = self.db.get_video_with_tags(video_id)
        
        # Print results
        print(f"\n✓ Successfully processed and stored in database!")
        print(f"  Video ID: {video_id}")
        print(f"  Tags: {', '.join(result['tags'][:10])}...")
        print(f"  Thumbnails: {len(thumbnails)} stored")
        if audio_result['has_speech']:
            print(f"  Audio: {audio_result['word_count']} words transcribed ({audio_result['language']})")
        
        return result
    
    def process_directory(self, directory: str, recursive: bool = False, force: bool = False):
        """
        Process all videos in a directory
        
        Args:
            directory: Directory path
            recursive: Search subdirectories
            force: Force reprocessing of all videos
        """
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        directory_path = Path(directory)
        
        if recursive:
            video_files = []
            for ext in video_extensions:
                video_files.extend(directory_path.rglob(f"*{ext}"))
        else:
            video_files = []
            for ext in video_extensions:
                video_files.extend(directory_path.glob(f"*{ext}"))
        
        print(f"\nFound {len(video_files)} video(s) to process")
        
        successful = 0
        failed = 0
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(video_files)}]")
            print(f"{'='*60}")
            
            try:
                result = self.process_video(str(video_file), force=force)
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing {video_file.name}: {e}")
                failed += 1
        
        print(f"\n{'='*60}")
        print("Batch processing complete!")
        print(f"{'='*60}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"\nDatabase: {self.db.db_path}")
        
        # Show statistics
        stats = self.db.get_statistics()
        print(f"\nDatabase Statistics:")
        print(f"  Total videos: {stats['total_videos']}")
        print(f"  Unique tags: {stats['unique_tags']}")
        print(f"  Total size: {stats['total_size_gb']} GB")
        print(f"  Total duration: {stats['total_duration_hours']} hours")
        if stats.get('videos_with_speech', 0) > 0:
            print(f"  Videos with speech: {stats['videos_with_speech']}")
            print(f"  Total words transcribed: {stats['total_words_transcribed']:,}")
    
    def search_by_tag(self, tag: str, limit: int = 50) -> List[Dict]:
        """Search for videos containing a specific tag"""
        return self.db.search_videos(tags=[tag.lower()], limit=limit)
    
    def show_statistics(self):
        """Display database statistics"""
        stats = self.db.get_statistics()
        
        print(f"\n{'='*60}")
        print("Video Archive Statistics")
        print(f"{'='*60}")
        print(f"Total videos: {stats['total_videos']}")
        print(f"Unique tags: {stats['unique_tags']}")
        print(f"Total storage: {stats['total_size_gb']} GB")
        print(f"Total duration: {stats['total_duration_hours']} hours")
        
        if stats.get('videos_with_speech', 0) > 0:
            print(f"Videos with speech: {stats['videos_with_speech']}")
            print(f"Total words: {stats['total_words_transcribed']:,}")
        
        # Show top tags
        print(f"\nTop 20 Tags:")
        tags = self.db.get_all_tags()[:20]
        for tag, count in tags:
            print(f"  {tag}: {count} videos")
    
    def cleanup(self):
        """Clean up resources"""
        self.analyzer.cleanup()
        if self.audio_analyzer:
            self.audio_analyzer.cleanup()
        self.db.close()


def main():
    """Main entry point"""
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
    warnings.filterwarnings("ignore", message="`resume_download` is deprecated", category=FutureWarning)

    parser = argparse.ArgumentParser(
        description="Automatically tag videos using multimodal AI and store in SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video with LLaVA (default)
  python video_tagger.py video.mp4
  
  # Process with audio transcription
  python video_tagger.py video.mp4 --audio
  
  # Use different model
  python video_tagger.py video.mp4 --model blip2
  
  # Process directory with German tags
  python video_tagger.py /path/to/videos --recursive --language de
  
  # Search for videos
  python video_tagger.py . --search outdoor
  
  # Show statistics
  python video_tagger.py . --stats
        """
    )
    parser.add_argument('input', nargs='?', default='.', help='Video file or directory')
    parser.add_argument('--frames', type=int, default=8, help='Number of frames to extract (default: 8)')
    parser.add_argument(
        '--model',
        choices=['llava', 'llava-large', 'blip2', 'instructblip'],
        default='llava',
        help='Multimodal AI model (default: llava)'
    )
    parser.add_argument('--language', type=str, default='en', help='Tag language (default: en)')
    parser.add_argument('--db', help='SQLite database path')
    parser.add_argument('--audio', action='store_true', help='Enable audio transcription')
    parser.add_argument('--whisper-model', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       default='base', help='Whisper model size (default: base)')
    parser.add_argument('--audio-language', type=str, help='Force audio transcription language')
    parser.add_argument('--recursive', action='store_true', help='Process subdirectories')
    parser.add_argument('--force', action='store_true', help='Force reprocessing')
    parser.add_argument('--search', help='Search videos by tag')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--config', help='Path to JSON configuration file')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from: {args.config}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config file: {e}")
    
    # Determine database path - CLI > config > auto
    db_path = args.db or config.get('database', {}).get('path')
    if db_path is None:
        if Path(args.input).is_dir():
            db_path = str(Path(args.input) / "video_archive.db")
        else:
            db_path = str(Path(args.input).parent / "video_archive.db")
    
    # Combine settings: CLI arguments override config file, which overrides defaults
    # Processing settings
    processing_config = config.get('processing', {})
    num_frames = args.frames if args.frames != 8 else processing_config.get('num_frames', 8)
    num_thumbnails = processing_config.get('num_thumbnails', 5)
    recursive = args.recursive or processing_config.get('recursive_search', False)
    force = args.force or processing_config.get('force_reprocess', False)
    
    # Model settings
    model_config = config.get('model', {})
    model_name = args.model if args.model != 'llava' else model_config.get('name', 'llava')
    
    # Tag settings
    tag_config = config.get('tags', {})
    tag_language = args.language if args.language != 'en' else tag_config.get('language', 'en')
    tag_stopwords = tag_config.get('stopwords', None)
    
    # Audio settings
    audio_config = config.get('audio', {})
    enable_audio = args.audio or audio_config.get('enabled', False)
    whisper_model = args.whisper_model if args.whisper_model != 'base' else audio_config.get('whisper_model', 'base')
    audio_language = args.audio_language or audio_config.get('language')
    no_pre_detect = audio_config.get('no_pre_detect', False)

    # Create tagger instance with simplified config (no more separate LLM settings!)
    tagger = VideoTagger(
        num_frames=num_frames,
        num_thumbnails=num_thumbnails,
        model_name=model_name,
        tag_language=tag_language,
        tag_stopwords=tag_stopwords,
        db_path=db_path,
        enable_audio=enable_audio,
        whisper_model=whisper_model,
        language=audio_language,
        no_pre_detect=no_pre_detect
    )
    
    try:
        if args.stats:
            tagger.show_statistics()
            return
        
        if args.search:
            results = tagger.search_by_tag(args.search)
            print(f"\nFound {len(results)} video(s) with tag '{args.search}':")
            print(f"{'='*60}")
            for video in results:
                print(f"\n{video['file_name']}")
                print(f"  Path: {video['file_path']}")
                print(f"  Duration: {video['duration_seconds']:.1f}s")
                video_id = video['id']
                full_info = tagger.db.get_video_with_tags(video_id)
                print(f"  Tags: {', '.join(full_info['tags'][:10])}...")
            return
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            tagger.process_video(str(input_path), force=force)
        elif input_path.is_dir():
            tagger.process_directory(str(input_path), recursive=recursive, force=force)
        else:
            print(f"Error: {args.input} is not a valid file or directory")
            return
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        tagger.cleanup()


if __name__ == "__main__":
    main()