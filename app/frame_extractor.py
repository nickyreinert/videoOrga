"""
Video Frame Extraction Module
Extracts representative frames from videos for AI analysis
"""

import cv2
import os
import base64
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from io import BytesIO
from PIL import Image


class FrameExtractor:
    """Extracts frames from videos using duration-based sampling"""
    
    def __init__(self, frames_per_minute: float = 2.0, min_frames: int = 3, max_frames: int = 50, num_thumbnails: int = 5):
        """
        Initialize frame extractor with duration-based scaling
        
        Args:
            frames_per_minute: Number of frames to extract per minute of video
            min_frames: Minimum number of frames to extract
            max_frames: Maximum number of frames to extract
            num_thumbnails: Number of thumbnail previews to extract
        """
        self.frames_per_minute = frames_per_minute
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.num_thumbnails = num_thumbnails
    
    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], dict]:
        """
        Extract frames uniformly distributed across the video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (list of frame arrays, video metadata dict)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata = {
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': duration,
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}",
            'codec': self._get_codec(cap)
        }
        
        # Calculate number of frames to extract based on video duration
        duration_minutes = duration / 60.0
        calculated_frames = int(np.ceil(duration_minutes * self.frames_per_minute))
        num_frames_to_extract = max(self.min_frames, min(calculated_frames, self.max_frames))
        
        print(f"Video duration: {duration:.1f}s ({duration_minutes:.2f} min)")
        print(f"Frames to extract: {num_frames_to_extract} (factor: {self.frames_per_minute}/min, range: {self.min_frames}-{self.max_frames})")
        
        # Calculate frame indices to extract (uniformly distributed)
        if total_frames <= num_frames_to_extract:
            # If video has fewer frames than requested, take all
            frame_indices = list(range(total_frames))
        else:
            # Distribute frames evenly, avoiding first 5% and last 10% (often black or credits)
            start_frame = int(total_frames * 0.05)
            end_frame = int(total_frames * 0.90)
            frame_indices = np.linspace(start_frame, end_frame, 
                                       num_frames_to_extract, dtype=int).tolist()
        
        # Extract frames
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB (OpenCV uses BGR by default)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                print(f"Warning: Could not read frame {idx}")
        
        cap.release()
        
        print(f"Extracted {len(frames)} frames from {Path(video_path).name}")
        return frames, metadata
    
    def _get_codec(self, cap) -> str:
        """Get video codec as string"""
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        return codec.strip() if codec.strip() else "unknown"
    
    def extract_thumbnails(self, video_path: str) -> List[Tuple]:
        """
        Extract random frames as thumbnails (base64 encoded)
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of tuples: (frame_number, base64_image_data, width, height)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Select random frames (avoid first 5% and last 10%)
        start_frame = int(total_frames * 0.05)
        end_frame = int(total_frames * 0.90)
        
        if end_frame - start_frame < self.num_thumbnails:
            # If video is too short, just use evenly spaced frames
            frame_indices = np.linspace(start_frame, end_frame, self.num_thumbnails, dtype=int).tolist()
        else:
            # Random selection
            frame_indices = sorted(random.sample(range(start_frame, end_frame), self.num_thumbnails))
        
        thumbnails = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize to thumbnail size (max 320px wide, maintaining aspect ratio)
                height, width = frame_rgb.shape[:2]
                max_width = 320
                
                if width > max_width:
                    scale = max_width / width
                    new_width = max_width
                    new_height = int(height * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                    width, height = new_width, new_height
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Encode to base64 JPEG
                buffer = BytesIO()
                pil_image.save(buffer, format="JPEG", quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                thumbnails.append((frame_idx, img_base64, width, height))
        
        cap.release()
        
        print(f"Extracted {len(thumbnails)} thumbnails")
        return thumbnails


def test_extraction(video_path: str):
    """Test function to verify frame extraction works"""
    extractor = FrameExtractor(frames_per_minute=2.0, min_frames=3, max_frames=50)
    
    try:
        frames, metadata = extractor.extract_frames(video_path)
        
        print("\n=== Video Metadata ===")
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
        print(f"\n=== Extracted Frames ===")
        print(f"Number of frames: {len(frames)}")
        if frames:
            print(f"Frame shape: {frames[0].shape}")
        
        return frames, metadata
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None, None


if __name__ == "__main__":
    # Test with a video file
    import sys
    
    if len(sys.argv) > 1:
        test_extraction(sys.argv[1])
    else:
        print("Usage: python frame_extractor.py <path_to_video>")
        print("Example: python frame_extractor.py sample_video.mp4")