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
                 summary_prompt_template: str = None,
                 summary_context_window: int = 512,  # IGNORED - kept for compatibility
                 stopwords: Optional[List[str]] = None):
        """
        Initialize AI analyzer with multimodal LLM
        
        Args:
            model_name: Multimodal model ('llava', 'llava-large', 'blip2', 'instructblip')
            device: Device to run on ('cuda', 'cpu', or 'auto')
            tag_language: Target language for tags (e.g., 'en', 'de', 'fr')
            summary_llm_model: IGNORED (kept for config compatibility)
            summary_prompt_template: Template for the summary prompt
            summary_context_window: IGNORED (kept for config compatibility)
            stopwords: Custom list of stopwords to remove from tags
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tag_language = tag_language.lower()
        self.model = None
        self.processor = None
        self.summary_prompt_template = summary_prompt_template
        
        # Ignore old config parameters but don't break if they're passed
        if summary_llm_model:
            print("Note: summary_llm_model is no longer needed with multimodal LLM")
        
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
        
        # 1. Load NLTK stopwords if available
        if NLTK_AVAILABLE:
            try:
                lang_map = {'en': 'english', 'de': 'german', 'fr': 'french', 'es': 'spanish'}
                if self.tag_language in lang_map:
                    self.stopwords.update(nltk_stopwords.words(lang_map[self.tag_language]))
                    print(f"Loaded {len(self.stopwords)} NLTK stopwords for language '{self.tag_language}'.")
            except Exception as e:
                print(f"Warning: Could not load NLTK stopwords for '{self.tag_language}': {e}")
        
        # 2. Add common verb stopwords (often missed by standard lists)
        COMMON_VERBS_EN = [
            'is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did',
            'can', 'could', 'should', 'would', 'will', 'shall', 'may', 'might',
            'must', 'got', 'getting', 'gotten', 'needs', 'need', 'like', 'likes',
            'going', 'want', 'wants', 'make', 'makes', 'made', 'see', 'sees', 'saw',
            'look', 'looks', 'looking'
        ]
        
        COMMON_VERBS_DE = [
            'ist', 'sind', 'war', 'waren', 'hat', 'haben', 'hatte', 'tun', 'tut',
            'täte', 'kann', 'konnte', 'soll', 'sollte', 'würde', 'wird', 'wurde',
            'muss', 'braucht', 'brauchen', 'möchte', 'möchten', 'gibt', 'geben',
            'geht', 'gehen', 'sieht', 'sehen', 'sah', 'macht', 'machen', 'gemacht',
            'lässt', 'lassen', 'kommt', 'kommen', 'kam', 'steht', 'stehen', 'stand',
            'liegt', 'liegen', 'lag', 'sitzt', 'sitzen', 'saß'
        ]
        
        if self.tag_language == 'en':
            self.stopwords.update(COMMON_VERBS_EN)
        elif self.tag_language == 'de':
            self.stopwords.update(COMMON_VERBS_DE)
            
        # 3. Merge custom stopwords
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
        
        lang_name = self.lang_names.get(self.tag_language, 'English')
        lang_instruction = f" strictly in {lang_name}" if self.tag_language != 'en' else ""

        prompt = f"""USER: <image>
Describe this video frame in one concise sentence{lang_instruction}.
ASSISTANT:"""

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
                    
        print(f"  Description samples: {descriptions[0][:20]}...")

        return {
            'descriptions': descriptions,
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
        # Strip trailing numbers in parentheses like "(1)", "(2)" before processing
        text = re.sub(r'\s*\(\d+\)\s*', ' ', text)
        
        # Keep letters, spaces, and international characters (unicode support)
        # Using [^\w\s] to remove punctuation but keep words
        text = re.sub(r'[^\w\s]+', ' ', text.lower())
        pre_tags = [tag for tag in text.split(' ')]
        
        # Filter out empty, short, and stopword tags
        tags = []
        for tag in pre_tags:
            # Basic filtering: length > 2
            if not tag or len(tag) <= 2:
                continue
            
            # Max length check (prevent malicious/corrupted tags)
            if len(tag) > 30:
                continue
                
            # Stopword filtering
            if tag in self.stopwords:
                continue
                
            # Check for repeated characters (e.g. "nnn")
            if len(set(tag)) == 1:
                continue
            
            # Detect repeated substring patterns (e.g., "taschentaschentasche...")
            if self._has_repeated_pattern(tag):
                continue
                
            tags.append(tag)
        
        return tags
    
    def clean_tag_list(self, tags: List[str]) -> List[str]:
        """
        Clean a list of existing tags using the configured filters
        
        Args:
            tags: List of tags to clean
            
        Returns:
            List of cleaned tags
        """
        cleaned_tags = []
        seen_tags = set()
        
        for tag in tags:
            # Normalize
            tag = tag.lower().strip()
            
            # Remove special chars (keep only letters and spaces)
            tag = re.sub(r'[^\w\s]+', '', tag)
            
            # Basic filtering: length > 2
            if not tag or len(tag) <= 2:
                continue
            
            # Max length check
            if len(tag) > 30:
                continue
                
            # Stopword filtering
            if tag in self.stopwords:
                continue
                
            # Check for repeated characters
            if len(set(tag)) == 1:
                continue
            
            # Detect repeated substring patterns
            if self._has_repeated_pattern(tag):
                continue
            
            # Deduplicate
            if tag in seen_tags:
                continue
                
            seen_tags.add(tag)
            cleaned_tags.append(tag)
        
        return cleaned_tags
    
    def _has_repeated_pattern(self, text: str) -> bool:
        """
        Detect if a string has a repeated substring pattern
        
        Args:
            text: String to check
            
        Returns:
            True if repeated pattern detected, False otherwise
        """
        # Check for patterns of length 3 to len(text)//2
        for pattern_len in range(3, len(text) // 2 + 1):
            pattern = text[:pattern_len]
            # Count how many times this pattern appears at the start
            count = 0
            pos = 0
            while pos < len(text) and text[pos:pos+pattern_len] == pattern:
                count += 1
                pos += pattern_len
            
            # If pattern repeats 3+ times and covers most of the string, it's suspicious
            if count >= 3 and pos >= len(text) * 0.7:
                return True
        
        return False

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
        lang_instruction = f" strictly in {lang_name}" if self.tag_language != 'en' else ""
        
        # Combine visual descriptions
        unique_descriptions = sorted(list(set(visual_descriptions)))
        visual_context = "\n".join(f"- {desc}" for desc in unique_descriptions[:10])
        
        # Truncate audio if too long
        if len(audio_transcript) > 1000:
            audio_transcript = audio_transcript[:1000] + "..."
        
        # Use configured prompt template if available, otherwise fallback (though fallback shouldn't happen with correct config)
        if self.summary_prompt_template:
            prompt = self.summary_prompt_template.format(
                language=lang_name,
                visual_context=visual_context,
                audio_transcript=audio_transcript
            )
        else:
            # Fallback prompt
            prompt = f"""USER: Summarize this video strictly in {lang_name}.
Visuals: {visual_context}
Audio: {audio_transcript}
Provide a concise summary paragraph strictly in {lang_name}.
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
        
        # Clean up output (remove prompt parts if they leak)
        # Note: This is harder with a custom prompt, but we can try to remove the prompt itself if it's echoed
        if prompt in summary:
            summary = summary.replace(prompt, "")
            
        # Also try to remove standard chat markers
        for marker in ["USER:", "ASSISTANT:", "[INST]", "[/INST]"]:
            summary = summary.replace(marker, "")
        
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
            
            # extract tags from summary
            tags = self._extract_tags_from_text(summary)
            
            tags = sorted(list(tags))[:20]
            
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