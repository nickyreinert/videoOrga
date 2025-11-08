"""
AI Frame Analyzer
Analyzes image frames using various models (BLIP, CLIP) and generates tags.
Includes translation and stopword removal capabilities.
"""

import torch
from PIL import Image
from typing import List, Dict, Optional
import re

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

class AIAnalyzer:
    """Analyzes image frames to generate descriptions and tags"""

    def __init__(self,
                 model_name: str = "blip",
                 device: str = "auto",
                 tag_language: str = 'en',
                 stopwords: Optional[List[str]] = None):
        """
        Initialize AI analyzer

        Args:
            model_name: AI model to use ('blip', 'blip2', 'clip')
            device: Device to run on ('cuda', 'cpu', or 'auto')
            tag_language: Target language for tags (e.g., 'en', 'de').
            stopwords: Custom list of stopwords to remove from tags.
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.processor = None
        self.tag_language = tag_language.lower()
        self.translator = None

        # Setup stopwords
        self.stopwords = set()
        if NLTK_AVAILABLE:
            try:
                lang_map = {'en': 'english', 'de': 'german', 'fr': 'french', 'es': 'spanish'}
                if self.tag_language in lang_map:
                    self.stopwords.update(nltk_stopwords.words(lang_map[self.tag_language]))
                    print(f"Loaded {len(self.stopwords)} stopwords from NLTK for language '{self.tag_language}'.")
            except Exception as e:
                print(f"Warning: Could not load NLTK stopwords for '{self.tag_language}': {e}")
        
        # Add custom stopwords from config if any are provided
        if stopwords:
            self.stopwords.update(stopwords)

        print(f"AI Analyzer initialized (model: {model_name}, device: {self.device})")
        if self.tag_language != 'en':
            print(f"Tag language set to: {self.tag_language.upper()}")
        if self.stopwords:
            print(f"Stopword removal enabled for tags ({len(self.stopwords)} words).")

    def _setup_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self):
        """Load the selected AI model and processor"""
        if self.model is not None:
            return

        print(f"Loading AI model ({self.model_name})...")
        from transformers import BlipProcessor, BlipForConditionalGeneration

        try:
            if self.model_name == 'blip':
                model_id = "Salesforce/blip-image-captioning-large"
                self.processor = BlipProcessor.from_pretrained(model_id)
                self.model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
            else:
                raise NotImplementedError(f"Model '{self.model_name}' is not yet supported.")
            print("AI model loaded successfully.")
        except Exception as e:
            print(f"Error loading model '{self.model_name}': {e}")
            raise

        # Load translation model if needed
        if self.tag_language != 'en':
            print(f"Loading translation model for '{self.tag_language.upper()}'...")
            from transformers import pipeline
            try:
                self.translator = pipeline("translation_en_to_" + self.tag_language, model=f"Helsinki-NLP/opus-mt-en-{self.tag_language}")
                print("Translation model loaded.")
            except Exception as e:
                print(f"Warning: Could not load translation model for '{self.tag_language}'. Tags will be in English. Error: {e}")
                self.translator = None

    def analyze_frames(self, frames: List[Image.Image]) -> Dict:
        """
        Analyze a list of frames to generate descriptions and tags.

        Args:
            frames: List of PIL Image objects.

        Returns:
            Dictionary with descriptions, tags, and frame count.
        """
        self.load_model()

        descriptions = []
        all_tags = set()

        for i, frame in enumerate(frames):
            print(f"  Analyzing frame {i+1}/{len(frames)}...")
            inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=50)
            description = self.processor.decode(out[0], skip_special_tokens=True)
            descriptions.append(description)

            # Translate if necessary
            if self.translator:
                try:
                    translated_description = self.translator(description)[0]['translation_text']
                except Exception as e:
                    print(f"Warning: Translation failed for a frame. Using English. Error: {e}")
                    translated_description = description
            else:
                translated_description = description

            # Extract tags from the (translated) description
            tags = self._extract_tags_from_text(translated_description)
            all_tags.update(tags)

        # Post-process tags
        final_tags = sorted(list(all_tags))

        return {
            'descriptions': descriptions,
            'tags': final_tags,
            'frame_count': len(frames)
        }

    def _extract_tags_from_text(self, text: str) -> List[str]:
        """
        Extracts and cleans keywords from a given text.

        Args:
            text: The text to process.

        Returns:
            A list of cleaned, lowercased tags.
        """
        # Remove "a photography of", "a picture of", etc.
        text = re.sub(r'^\s*(a photograph of|a picture of|a close up of|an image of)\s*', '', text, flags=re.IGNORECASE)
        # Split into words, remove non-alphanumeric characters
        words = re.findall(r'\b[a-zA-Z\u00C0-\u017F]+\b', text.lower())

        # Filter out stopwords and single-character words
        if self.stopwords:
            tags = [word for word in words if word not in self.stopwords and len(word) > 2]
        else:
            tags = [word for word in words if len(word) > 2]

        return tags

    def cleanup(self):
        """Free up GPU memory"""
        del self.model
        del self.processor
        if self.translator:
            del self.translator
        self.model = None
        self.processor = None
        self.translator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("AI analyzer cleaned up.")