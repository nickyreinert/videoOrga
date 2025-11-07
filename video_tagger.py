"""
Video Auto-Tagger - Main Script with SQLite Storage
Processes videos, extracts frames, analyzes with AI, and stores in SQLite database
"""

import os
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
                 model_name: str = "blip",
                 db_path: str = None,
                 enable_audio: bool = False,
                 whisper_model: str = "base",
                 language: str = None,
                 no_pre_detect: bool = False):
        """
        Initialize video tagger with SQLite backend
        
        Args:
            num_frames: Number of frames to extract per video
            model_name: AI model to use ('blip2', 'blip', 'clip')
            db_path: Path to SQLite database (default: video_archive.db in current dir)
            enable_audio: Whether to transcribe and summarize audio
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            language: Language for transcription ('en', 'de', etc.) or None for auto-detect
            no_pre_detect: Disable language pre-detection when multiple languages are given
        """
        self.extractor = FrameExtractor(num_frames=num_frames)
        self.analyzer = AIAnalyzer(model_name=model_name)
        
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
        
        # Set database path - if not specified, use same directory as input or current dir
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
        print("\n[3/6] Extracting thumbnail previews...")
        try:
            thumbnails = self.extractor.extract_thumbnails(video_path, num_thumbnails=3)
        except Exception as e:
            print(f"Warning: Could not extract thumbnails: {e}")
            thumbnails = []
        
        # Step 4: Analyze frames with AI
        print("\n[4/6] Analyzing frames with AI...")
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
            print("\n[5/6] Analyzing audio (transcription & summary)...")
            try:
                audio_result = self.audio_analyzer.analyze_video_audio(
                    video_path,
                    summarize=True
                )
            except Exception as e:
                print(f"Warning: Audio analysis failed: {e}")
        else:
            print("\n[5/6] Skipping audio analysis (use --audio to enable)")
        
        # Step 6: Store everything in database
        print("\n[6/6] Storing in database...")
        
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
        print(f"  Tags: {', '.join(result['tags'])}")
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
        """
        Search for videos containing a specific tag
        
        Args:
            tag: Tag to search for
            limit: Maximum results
            
        Returns:
            List of matching video metadata
        """
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
    # Suppress the specific UserWarning from PyTorch about TypedStorage
    # This is a known issue in certain versions of torch/transformers and is safe to ignore
    warnings.filterwarnings(
        "ignore",
        message="TypedStorage is deprecated"
    )

    # Suppress the FutureWarning from huggingface_hub about resume_download
    warnings.filterwarnings(
        "ignore",
        message="`resume_download` is deprecated",
        category=FutureWarning
    )

    parser = argparse.ArgumentParser(
        description="Automatically tag videos using AI and store in SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python video_tagger.py video.mp4
  
  # Process with audio transcription
  python video_tagger.py video.mp4 --audio
  
  # Process directory (creates video_archive.db in current dir)
  python video_tagger.py /path/to/videos --recursive
  
  # Use custom database location
  python video_tagger.py video.mp4 --db /path/to/archive.db
  
  # Search for videos
  python video_tagger.py . --search outdoor
  
  # Show statistics
  python video_tagger.py . --stats
        """
    )
    parser.add_argument(
        'input',
        nargs='?',
        default='.',
        help='Video file or directory to process (default: current directory)'
    )
    parser.add_argument(
        '--frames',
        type=int,
        default=8,
        help='Number of frames to extract per video (default: 8)'
    )
    parser.add_argument(
        '--model',
        choices=['blip', 'blip2', 'clip'],
        default='blip',
        help='AI model to use (default: blip)'
    )
    parser.add_argument(
        '--db',
        help='SQLite database path (default: video_archive.db in current directory)'
    )
    parser.add_argument(
        '--audio',
        action='store_true',
        help='Enable audio transcription and summarization (requires Whisper)'
    )
    parser.add_argument(
        '--whisper-model',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='base',
        help='Whisper model size for transcription (default: base)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='de,en',
        help='Force transcription language(s) (e.g., "en" or "de,en"). Skips auto-detection if one language is given, otherwise pre-detects within the list.'
    )
    parser.add_argument(
        '--no-language-pre-detect',
        action='store_true',
        help='Disable pre-detection when multiple languages are provided. Uses the first language in the list.'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Process subdirectories recursively'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing of already tagged videos'
    )
    parser.add_argument(
        '--search',
        help='Search videos by tag'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics'
    )
    
    args = parser.parse_args()
    
    # Determine database path
    db_path = args.db
    if db_path is None:
        # If processing a directory, put DB in that directory
        if Path(args.input).is_dir():
            db_path = str(Path(args.input) / "video_archive.db")
        else:
            # If processing a file, put DB in same directory as file
            db_path = str(Path(args.input).parent / "video_archive.db")
    
    # Create tagger instance
    tagger = VideoTagger(
        num_frames=args.frames,
        model_name=args.model,
        db_path=db_path,
        enable_audio=args.audio,
        whisper_model=args.whisper_model,
        language=args.language,
        no_pre_detect=args.no_language_pre_detect
    )
    
    try:
        # Statistics mode
        if args.stats:
            tagger.show_statistics()
            return
        
        # Search mode
        if args.search:
            results = tagger.search_by_tag(args.search)
            print(f"\nFound {len(results)} video(s) with tag '{args.search}':")
            print(f"{'='*60}")
            for video in results:
                print(f"\n{video['file_name']}")
                print(f"  Path: {video['file_path']}")
                print(f"  Duration: {video['duration_seconds']:.1f}s")
                print(f"  Date: {video['parsed_datetime'] or video['file_modified_date']}")
                
                # Get tags
                video_id = video['id']
                full_info = tagger.db.get_video_with_tags(video_id)
                print(f"  Tags: {', '.join(full_info['tags'])}")
            return
        
        # Check if input is file or directory
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Process single video
            tagger.process_video(str(input_path), force=args.force)
        elif input_path.is_dir():
            # Process directory
            tagger.process_directory(str(input_path), recursive=args.recursive, force=args.force)
        else:
            print(f"Error: {args.input} is not a valid file or directory")
            return
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        tagger.cleanup()


if __name__ == "__main__":
    main()