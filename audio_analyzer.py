"""
Audio Analysis Module
Extracts audio, transcribes with Whisper, and summarizes with local LLM
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from tqdm import tqdm

class AudioAnalyzer:
    """Analyzes audio from videos: transcription and summarization"""
    
    def __init__(self, 
                 whisper_model: str = "base",
                 device: str = "auto",
                 language: str = None,
                 no_pre_detect: bool = False):
        """
        Initialize audio analyzer
        
        Args:
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
                          tiny: fastest, least accurate (~1GB VRAM)
                          base: good balance (~1GB VRAM) ⭐ RECOMMENDED
                          small: better quality (~2GB VRAM)
                          medium: high quality (~5GB VRAM)
                          large: best quality (~10GB VRAM)
            device: Device to run on ('cuda', 'cpu', or 'auto')
            language: Language code (e.g., 'en', 'de', 'es') or None for auto-detect
            no_pre_detect: Disable language pre-detection when multiple languages are given
        """
        self.whisper_model_name = whisper_model
        self.language = language
        self.no_pre_detect = no_pre_detect
        self.device = self._setup_device(device)
        self.whisper_model = None
        self.summarizer = None
        
        print(f"Audio analyzer initialized (device: {self.device})")
    
    def _setup_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_whisper(self):
        """Load Whisper model for transcription"""
        if self.whisper_model is not None:
            return
        
        print(f"Loading Whisper model ({self.whisper_model_name})...")
        
        try:
            import whisper
            
            self.whisper_model = whisper.load_model(
                self.whisper_model_name,
                device=self.device
            )
            
            print(f"Whisper model loaded successfully")
            
        except ImportError:
            print("Whisper not installed. Install with: pip install openai-whisper")
            raise
        except Exception as e:
            print(f"Error loading Whisper: {e}")
            raise
    
    def load_summarizer(self):
        """Load local LLM for summarization (using transformers)"""
        if self.summarizer is not None:
            return
        
        print("Loading summarization model...")
        
        try:
            from transformers import pipeline
            
            # Use a small, efficient summarization model
            # Options: 
            # - "facebook/bart-large-cnn" (good quality, ~1.6GB)
            # - "sshleifer/distilbart-cnn-12-6" (faster, smaller, ~1GB) ⭐ RECOMMENDED
            # - "google/pegasus-xsum" (very good, but larger)
            
            model_name = "sshleifer/distilbart-cnn-12-6"
            
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0 if self.device == "cuda" else -1
            )
            
            print("Summarization model loaded successfully")
            
        except Exception as e:
            print(f"Error loading summarizer: {e}")
            print("Note: Summarization will be skipped if model can't load")
            self.summarizer = None
    
    def extract_audio(self, video_path: str, audio_path: str = None) -> str:
        """
        Extract audio from video using FFmpeg
        
        Args:
            video_path: Path to video file
            audio_path: Optional output path for audio (default: temp file)
            
        Returns:
            Path to extracted audio file (WAV format)
        """
        if audio_path is None:
            # Create temporary audio file
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(
                temp_dir,
                f"audio_{Path(video_path).stem}.wav"
            )
        
        # Use FFmpeg to extract audio
        # -vn: no video
        # -acodec pcm_s16le: PCM 16-bit audio
        # -ar 16000: 16kHz sample rate (Whisper's preferred rate)
        # -ac 1: mono audio
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',  # Overwrite output file
            audio_path
        ]
        
        try:
            # Run FFmpeg silently
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                return audio_path
            else:
                raise Exception("Audio file was not created or is empty")
                
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg failed to extract audio: {e}")
        except FileNotFoundError:
            raise Exception("FFmpeg not found. Install with: apt install ffmpeg (Linux) or brew install ffmpeg (Mac)")
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio using Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        self.load_whisper()
        
        try:
            # If a list of languages is provided, perform pre-detection
            transcribe_language = self.language
            if self.language and ',' in self.language and not self.no_pre_detect:
                allowed_languages = [lang.strip() for lang in self.language.split(',')]
                print(f"  Pre-detecting language within: {allowed_languages}...")
                
                # Load the audio and detect the language
                import whisper
                audio = whisper.load_audio(audio_path)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
                _, probs = self.whisper_model.detect_language(mel)
                detected_lang = max(probs, key=probs.get)
                
                # If detected language is allowed, use it. Otherwise, use the first one as default.
                transcribe_language = detected_lang if detected_lang in allowed_languages else allowed_languages[0]
                print(f"  Detected '{detected_lang}', transcribing with '{transcribe_language}'")
            elif self.language and ',' in self.language and self.no_pre_detect:
                # Use the first language from the list without pre-detection
                transcribe_language = self.language.split(',')[0].strip()
                print(f"  Pre-detection disabled, using first language: '{transcribe_language}'")

            print(f"Transcribing audio...")
            # Let Whisper manage the progress bar by setting verbose=None.
            # This uses tqdm if it's installed and is more reliable than a custom callback.
            result = self.whisper_model.transcribe(
                audio_path, language=transcribe_language, fp16=(self.device == 'cuda'),
                verbose=None
            )
            # Extract text and metadata
            text = result['text'].strip()
            language = result.get('language', 'unknown')
            
            # Get segments (timestamped text chunks)
            segments = []
            for seg in result.get('segments', []):
                segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'].strip()
                })
            
            print(f"Transcription complete ({len(text)} chars, language: {language})")
            
            return {
                'text': text,
                'language': language,
                'segments': segments,
                'has_speech': len(text) > 0
            }
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'segments': [],
                'has_speech': False,
                'error': str(e)
            }
    
    def summarize_text(self, text: str, max_length: Optional[int] = None, min_length: int = 30) -> str:
        """
        Summarize text using local LLM
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in tokens (default: input_length/2)
            min_length: Minimum summary length in tokens
            
        Returns:
            Summary text
        """
        if not text or len(text.split()) < 20:
            return text  # Too short to summarize
        
        try:
            self.load_summarizer()
            
            if self.summarizer is None:
                print("Summarizer not available, returning original text")
                return text[:500] + "..." if len(text) > 500 else text
            
            print("Generating summary...")
            
            # Calculate appropriate max_length if not provided
            input_length = len(text.split())
            if max_length is None:
                max_length = max(min_length, input_length // 2)  # Target 50% reduction
            
            # Generate summary, letting the pipeline handle truncation
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True  # Ensure input is truncated to model's max length
            )
            
            summary_text = summary[0]['summary_text']
            print(f"Summary generated ({len(summary_text)} chars)")
            
            return summary_text
            
        except Exception as e:
            print(f"Summarization failed: {e}")
            # Return truncated original text as fallback
            return text[:500] + "..." if len(text) > 500 else text
    
    def analyze_video_audio(self, 
                           video_path: str,
                           summarize: bool = True,
                           cleanup_audio: bool = True) -> Dict:
        """
        Complete audio analysis pipeline: extract, transcribe, summarize
        
        Args:
            video_path: Path to video file
            summarize: Whether to generate summary
            cleanup_audio: Whether to delete temporary audio file
            
        Returns:
            Dictionary with all audio analysis results
        """
        print("\n" + "="*60)
        print("Audio Analysis")
        print("="*60)
        
        audio_path = None
        
        try:
            # Step 1: Extract audio
            print("\n[1/3] Extracting audio from video...")
            audio_path = self.extract_audio(video_path)
            audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            print(f"  Audio extracted: {audio_size_mb:.2f} MB")
            
            # Step 2: Transcribe
            print("\n[2/3] Transcribing audio...")
            transcription = self.transcribe_audio(audio_path)
            
            if not transcription['has_speech']:
                print("  No speech detected in video")
                return {
                    'has_speech': False,
                    'transcript': '',
                    'summary': '',
                    'language': 'none',
                    'word_count': 0
                }
            
            # Step 3: Summarize (optional)
            summary = ''
            if summarize and transcription['text']:
                print("\n[3/3] Generating summary...")
                summary = self.summarize_text(transcription['text'])
            else:
                print("\n[3/3] Skipping summary generation")
            
            # Compile results
            word_count = len(transcription['text'].split())
            
            result = {
                'has_speech': True,
                'transcript': transcription['text'],
                'summary': summary,
                'language': transcription['language'],
                'word_count': word_count,
                'segments': transcription['segments']
            }
            
            print("\n✓ Audio analysis complete")
            print(f"  Language: {result['language']}")
            print(f"  Words: {result['word_count']}")
            print(f"  Transcript: {len(result['transcript'])} chars")
            if summary:
                print(f"  Summary: {len(result['summary'])} chars")
            
            return result
            
        except Exception as e:
            print(f"\n✗ Audio analysis failed: {e}")
            return {
                'has_speech': False,
                'transcript': '',
                'summary': '',
                'language': 'unknown',
                'word_count': 0,
                'error': str(e)
            }
        
        finally:
            # Cleanup temporary audio file
            if cleanup_audio and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print("  Cleaned up temporary audio file")
                except:
                    pass
    
    def cleanup(self):
        """Free up GPU memory"""
        if self.whisper_model is not None:
            del self.whisper_model
            self.whisper_model = None
        
        if self.summarizer is not None:
            del self.summarizer
            self.summarizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Audio analyzer cleaned up")


def test_audio_analysis(video_path: str):
    """Test function to verify audio analysis works"""
    analyzer = AudioAnalyzer(
        whisper_model="base",  # Good balance
        device="auto"
    )
    
    try:
        result = analyzer.analyze_video_audio(video_path, summarize=True)
        
        print("\n" + "="*60)
        print("Results:")
        print("="*60)
        
        if result['has_speech']:
            print(f"\nLanguage: {result['language']}")
            print(f"Word Count: {result['word_count']}")
            
            print(f"\nTranscript:")
            print(result['transcript'][:500])
            if len(result['transcript']) > 500:
                print("...")
            
            if result['summary']:
                print(f"\nSummary:")
                print(result['summary'])
        else:
            print("\nNo speech detected in video")
            if 'error' in result:
                print(f"Error: {result['error']}")
        
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_audio_analysis(sys.argv[1])
    else:
        print("Usage: python audio_analyzer.py <path_to_video>")
        print("Example: python audio_analyzer.py sample_video.mp4")