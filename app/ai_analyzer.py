"""
AI Frame Analyzer - Refactored to use single multimodal LLM
Replaces BLIP + Translator + Summary LLM with one efficient model
COMPATIBLE WITH EXISTING CONFIG STRUCTURE
"""

import torch
from PIL import Image
from typing import List, Dict, Optional
import re
import os

# Silence tokenizer parallelism warnings globally
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

class AIAnalyzer:
    """Analyzes image frames and generates summaries using a multimodal LLM"""

    def __init__(self,
                 model_name: str = "llava",
                 device: str = "auto",
                 tag_language: str = 'en',
                 summary_llm_model: str = None,  # IGNORED - kept for compatibility
                 summary_prompt_template: str = None,  # IGNORED - kept for compatibility
                 summary_context_window: int = 512,  # IGNORED - kept for compatibility
                 stopwords: Optional[List[str]] = None):
        """
        Initialize AI analyzer with multimodal LLM
        
        Args:
            model_name: Multimodal model ('llava', 'llava-large', 'blip2', 'instructblip')
            device: Device to run on ('cuda', 'cpu', or 'auto')
            tag_language: Target language for tags (e.g., 'en', 'de', 'fr')
            summary_llm_model: IGNORED (kept for config compatibility)
            summary_prompt_template: IGNORED (kept for config compatibility)
            summary_context_window: IGNORED (kept for config compatibility)
            stopwords: Custom list of stopwords to remove from tags
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tag_language = tag_language.lower()
        self.model = None
        self.processor = None
        
        # Ignore old config parameters but don't break if they're passed
        if summary_llm_model or summary_prompt_template:
            print("Note: summary_llm_model and summary_prompt_template are no longer needed with multimodal LLM")
        
        # Map model names to HuggingFace model IDs
        self.model_mapping = {
            'llava': 'llava-hf/llava-1.5-7b-hf',
            'llava-large': 'llava-hf/llava-1.5-13b-hf',
            'blip2': 'Salesforce/blip2-opt-2.7b',
            'instructblip': 'Salesforce/instructblip-vicuna-7b',
            'blip': 'llava-hf/llava-1.5-7b-hf',  # Default to llava
            'clip': 'llava-hf/llava-1.5-7b-hf',
        }
        
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
        
        if stopwords:
            self.stopwords.update(stopwords)
        
        # Language name mapping for prompts
        self.lang_names = {
            'en': 'English',
            'de': 'German',
            'fr': 'French',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'pl': 'Polish'
        }
        
        print(f"AI Analyzer initialized (model: {model_name}, device: {self.device})")
        if self.tag_language != 'en':
            print(f"Tag language set to: {self.tag_language.upper()}")
        if self.stopwords:
            print(f"Stopword removal enabled ({len(self.stopwords)} words)")

    def _setup_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self):
        """Load the multimodal LLM model"""
        if self.model is not None:
            return

        model_id = self.model_mapping.get(self.model_name)
        if not model_id:
            print(f"Warning: Unknown model '{self.model_name}', defaulting to llava")
            model_id = self.model_mapping['llava']

        print(f"Loading multimodal model ({model_id})...")
        
        # Use new API to avoid deprecation warning
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        
        try:
            # Load processor with fast tokenizer
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                use_fast=True  # Silence slow processor warning
            )
            
            try:
                # Try loading with 4-bit quantization first
                print("Attempting to load model with 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    dtype=torch.float16,
                )
                print("Model loaded successfully with 4-bit quantization.")
            except Exception as e_4bit:
                print(f"Warning: 4-bit quantization failed: {e_4bit}. Falling back to 8-bit.")
                try:
                    # Fallback to 8-bit quantization
                    quantization_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_id,
                        quantization_config=quantization_config_8bit,
                        device_map="auto",
                        dtype=torch.float16,
                    )
                    print("Model loaded successfully with 8-bit quantization.")
                except Exception as e_8bit:
                    print(f"Warning: 8-bit quantization also failed: {e_8bit}. Falling back to default loading.")
                    # If 8-bit also fails, raise the exception to be caught by the outer block
                    raise e_8bit

        except Exception as e:
            print(f"Warning: Quantized loading failed: {e}. Falling back to default loading (this will use more memory).")
            # Fallback to loading without any quantization, then manually move to device.
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch.float16 if self.device == 'cuda' else torch.float32
            ).to(self.device)

            print("Model loaded successfully without quantization.")

        if self.model:
            self.device = next(self.model.parameters()).device
            print(f"Model is on device: {self.device}")

    def _build_prompt(self, task: str) -> str:
        """Build prompt for specific task"""
        lang_name = self.lang_names.get(self.tag_language, 'English')
        lang_instruction = f" in {lang_name}" if self.tag_language != 'en' else ""
        
        prompts = {
            'tags': f"""USER: <image>
Generate relevant tags for this video frame{lang_instruction}.
Output ONLY comma-separated keywords describing: main subjects, actions, setting, objects, colors, mood.
Do not include any other text.
ASSISTANT: Tags:""",
            
            'caption': f"""USER: <image>
Describe this video frame in one concise sentence{lang_instruction}.
ASSISTANT:""",
        }
        
        return prompts.get(task, prompts['tags'])

    def analyze_frame(self, image: Image.Image, task: str = 'tags') -> str:
        """
        Analyze a single frame
        
        Args:
            image: PIL Image object
            task: 'tags', 'caption', or 'detailed'
            
        Returns:
            Analysis result as string
        """
        self.load_model()
        
        prompt = self._build_prompt(task)

        print(f"Prompt for task '{task}': {prompt}...")
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200 if task == 'detailed' else 100,
                do_sample=False
            )
        
        # Decode only the newly generated tokens
        input_token_len = inputs.input_ids.shape[1]
        result = self.processor.decode(outputs[0][input_token_len:], skip_special_tokens=True)
                
        # The result is already clean, but we can strip just in case
        return result.strip()

    def analyze_frames(self, frames: List[Image.Image]) -> Dict:
        """
        Analyze multiple frames to generate descriptions and tags
        
        Args:
            frames: List of PIL Image objects
            
        Returns:
            Dictionary with descriptions, tags, and frame count
        """
        self.load_model()
        
        descriptions = []
        all_tags = set()
        
        print(f"Analyzing {len(frames)} frames in language '{self.tag_language}'...")
        for i, frame in enumerate(frames):
            print(f"  Analyzing frame {i+1}/{len(frames)}...")
            
            # Get caption for this frame
            caption = self.analyze_frame(frame, task='caption')
            descriptions.append(caption)
            
            # Get tags for this frame
            tags_str = self.analyze_frame(frame, task='tags')
            
            # Parse tags
            tags = self._extract_tags_from_text(tags_str)
            all_tags.update(tags)
        
        # Post-process tags
        final_tags = sorted(list(all_tags))
        
        print(f"  Generated {len(final_tags)} unique tags")
        print(f"  Sample tags: {', '.join(final_tags[:5])}...")
        print(f"  Description samples: {descriptions[0][:20]}...")

        return {
            'descriptions': descriptions,
            'tags': final_tags,
            'frame_count': len(frames)
        }

    def _extract_tags_from_text(self, text: str) -> List[str]:
        """
        Extract and clean tags from text
        
        Args:
            text: Raw text containing tags
            
        Returns:
            List of cleaned tags
        """
        # Remove common prefixes
        text = re.sub(r'^\s*(tags:|keywords:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*(a photograph of|a picture of|a close up of|an image of)\s*', '', text, flags=re.IGNORECASE)
        
        # Split by comma and clean
        tags = [tag.strip().lower() for tag in text.split(',')]
        
        # Also extract words if format isn't comma-separated
        if len(tags) <= 1:
            words = re.findall(r'\b[a-zA-Z\u00C0-\u017F]+\b', text.lower())
            tags = words
        
        # Filter out empty, short, and stopword tags
        tags = [
            tag for tag in tags 
            if tag and len(tag) > 2 and tag not in self.stopwords
        ]
        
        return tags

    def generate_video_summary(self, 
                               visual_descriptions: List[str],
                               audio_transcript: str = "") -> str:
        """
        Generate a video-level summary from frame descriptions and audio
        
        Args:
            visual_descriptions: List of frame descriptions
            audio_transcript: Audio transcript (optional)
            
        Returns:
            Summary text
        """
        self.load_model()
        
        lang_name = self.lang_names.get(self.tag_language, 'English')
        lang_instruction = f" in {lang_name}" if self.tag_language != 'en' else ""
        
        # Combine visual descriptions
        unique_descriptions = sorted(list(set(visual_descriptions)))
        visual_context = "\n".join(f"- {desc}" for desc in unique_descriptions[:10])
        
        # Truncate audio if too long
        if len(audio_transcript) > 1000:
            audio_transcript = audio_transcript[:1000] + "..."
        
        # Build prompt
        audio_section = f"\n\nAudio transcript:\n{audio_transcript}" if audio_transcript else ""
        
        prompt = f"""USER: Summarize this video{lang_instruction}.

The video shows:
{visual_context}{audio_section}

Provide a concise summary paragraph.
ASSISTANT:"""
        
        # Generate summary
        inputs = self.processor(
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7
            )
        
        summary = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up output
        for prompt_part in [prompt, "USER:", "ASSISTANT:"]:
            summary = summary.replace(prompt_part, "")
        
        return summary.strip()

    def generate_ai_summary_and_tags(self,
                                     visual_descriptions: List[str],
                                     audio_transcript: str,
                                     language: str) -> Optional[Dict]:
        """
        Generate consolidated summary and tags
        (Compatible interface with old system)
        
        Args:
            visual_descriptions: List of frame descriptions
            audio_transcript: Audio transcript
            language: Target language (uses self.tag_language instead)
            
        Returns:
            Dictionary with 'summary' and 'tags'
        """
        print("  Generating AI summary and tags with multimodal LLM...")
        
        try:
            # Generate summary
            summary = self.generate_video_summary(visual_descriptions, audio_transcript)
            
            # Extract tags from all descriptions
            all_tags = set()
            for desc in visual_descriptions:
                tags = self._extract_tags_from_text(desc)
                all_tags.update(tags)
            
            # If we have audio, extract keywords from it too
            if audio_transcript:
                words = re.findall(r'\b[a-zA-Z]{4,}\b', audio_transcript.lower())
                audio_tags = [w for w in words if w not in self.stopwords][:10]
                all_tags.update(audio_tags)
            
            tags = sorted(list(all_tags))[:20]
            
            print(f"  Generated summary ({len(summary)} chars)")
            print(f"  Generated {len(tags)} tags")
            
            return {
                'summary': summary,
                'tags': tags
            }
            
        except Exception as e:
            print(f"Error generating AI summary: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Legacy compatibility methods (no-op)
    def load_summary_generator(self):
        """Legacy method - no longer needed"""
        pass
    
    def cleanup(self):
        """Free up GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("AI analyzer cleaned up")