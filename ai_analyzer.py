"""
AI Image Analyzer Module
Uses local vision-language models to generate tags from video frames
Optimized for NVIDIA RTX 3070 (8GB VRAM)
"""

import torch
from PIL import Image
import numpy as np
from typing import List, Dict
from collections import Counter


class AIAnalyzer:
    """Analyzes images using local AI models to generate descriptive tags"""
    
    def __init__(self, model_name: str = "blip2", device: str = "auto"):
        """
        Initialize AI analyzer
        
        Args:
            model_name: Model to use ('blip2', 'blip', 'clip')
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.processor = None
        
        print(f"Using device: {self.device}")
        
    def _setup_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self):
        """Load the AI model (lazy loading to save memory)"""
        if self.model is not None:
            return
        
        print(f"Loading {self.model_name} model...")
        
        if self.model_name == "blip2":
            self._load_blip2()
        elif self.model_name == "blip":
            self._load_blip()
        elif self.model_name == "clip":
            self._load_clip()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _load_blip2(self):
        """Load BLIP-2 model (best for descriptive captions)"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            # Use the smaller BLIP-2 variant for 8GB VRAM
            model_id = "Salesforce/blip2-opt-2.7b"
            
            self.processor = Blip2Processor.from_pretrained(model_id, use_fast=True)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            
            print("BLIP-2 model loaded successfully")
            
        except Exception as e:
            print(f"Error loading BLIP-2: {e}")
            print("Install with: pip install transformers salesforce-lavis")
            raise
    
    def _load_blip(self):
        """Load BLIP model (faster, less VRAM)"""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            model_id = "Salesforce/blip-image-captioning-large"
            
            self.processor = BlipProcessor.from_pretrained(model_id, use_fast=True)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            print("BLIP model loaded successfully")
            
        except Exception as e:
            print(f"Error loading BLIP: {e}")
            raise
    
    def _load_clip(self):
        """Load CLIP model (fastest, good for classification)"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            model_id = "openai/clip-vit-large-patch14"
            
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id).to(self.device)
            
            print("CLIP model loaded successfully")
            
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            raise
    
    def analyze_frame(self, frame: np.ndarray) -> str:
        """
        Analyze a single frame and return a description
        
        Args:
            frame: Image as numpy array (RGB)
            
        Returns:
            Text description of the frame
        """
        self.load_model()
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        
        if self.model_name in ["blip2", "blip"]:
            return self._caption_with_blip(image)
        elif self.model_name == "clip":
            return self._classify_with_clip(image)
    
    def _caption_with_blip(self, image: Image) -> str:
        """Generate caption using BLIP/BLIP-2"""
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device, 
            torch.float16 if self.device == "cuda" else torch.float32
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_length=50)
        
        caption = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0].strip()
        
        return caption
    
    def _classify_with_clip(self, image: Image) -> str:
        """Classify image using CLIP with predefined categories"""
        # Common video content categories
        categories = [
            "indoor scene", "outdoor scene", "nature", "urban environment",
            "people", "food", "table", "kitchen", "living room", "bedroom",
            "car", "street", "highway", "autobahn", "traffic",
            "restaurant", "office", "store", "shopping",
            "daytime", "nighttime", "sunny", "cloudy",
            "close-up", "wide shot", "aerial view"
        ]
        
        inputs = self.processor(
            text=categories,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get top 5 categories
        top_probs, top_indices = probs[0].topk(5)
        results = [categories[idx] for idx in top_indices]
        
        return ", ".join(results)
    
    def analyze_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze multiple frames and aggregate results
        
        Args:
            frames: List of frame arrays
            
        Returns:
            Dictionary with tags and metadata
        """
        print(f"Analyzing {len(frames)} frames...")
        
        descriptions = []
        for i, frame in enumerate(frames):
            print(f"  Processing frame {i+1}/{len(frames)}...", end="\r")
            desc = self.analyze_frame(frame)
            descriptions.append(desc)
        
        print()  # New line after progress
        
        # Extract tags from descriptions
        tags = self._extract_tags_from_descriptions(descriptions)
        
        return {
            'descriptions': descriptions,
            'tags': tags,
            'frame_count': len(frames)
        }
    
    def _extract_tags_from_descriptions(self, descriptions: List[str]) -> List[str]:
        """
        Extract common tags/keywords from multiple descriptions
        
        Args:
            descriptions: List of text descriptions
            
        Returns:
            List of extracted tags
        """
        # Simple keyword extraction - count word frequency
        all_words = []
        
        # Words to ignore (common articles, prepositions, etc.)
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from',
            'this', 'that', 'these', 'those', 'and', 'or', 'but'
        }
        
        for desc in descriptions:
            # Simple word tokenization
            words = desc.lower().replace(',', ' ').split()
            words = [w.strip('.,!?;:()[]{}') for w in words]
            words = [w for w in words if len(w) > 2 and w not in stop_words]
            all_words.extend(words)
        
        # Count word frequency
        word_counts = Counter(all_words)
        
        # Get most common words as tags (appearing in at least 2 frames or top 10)
        min_frequency = max(2, len(descriptions) // 4)
        tags = [word for word, count in word_counts.most_common(15) 
                if count >= min_frequency]
        
        return tags[:10]  # Return top 10 tags
    
    def cleanup(self):
        """Free up GPU memory"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Model cleaned up")


def test_analyzer(image_path: str = None):
    """Test the analyzer with a sample image"""
    analyzer = AIAnalyzer(model_name="blip")  # Start with faster BLIP
    
    if image_path:
        from PIL import Image
        img = Image.open(image_path)
        frame = np.array(img)
        
        result = analyzer.analyze_frame(frame)
        print(f"\nAnalysis result: {result}")
    else:
        print("No image provided for testing")
        print("Usage: python ai_analyzer.py <path_to_image>")
    
    analyzer.cleanup()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_analyzer(sys.argv[1])
    else:
        test_analyzer()