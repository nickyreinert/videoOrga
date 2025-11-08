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
                 summary_llm_model: str = "google/flan-t5-base",
                 summary_prompt_template: str = None,
                 summary_context_window: int = 512,
                 stopwords: Optional[List[str]] = None):

        """
        Initialize AI analyzer

        Args:
            model_name: AI model to use ('blip', 'blip2', 'clip')
            device: Device to run on ('cuda', 'cpu', or 'auto')
            tag_language: Target language for tags (e.g., 'en', 'de').
            summary_llm_model: The model to use for generating summaries and tags.
            summary_prompt_template: The prompt template for the summary LLM.
            summary_context_window: The context window size for the summarizer model.
            stopwords: Custom list of stopwords to remove from tags.
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.processor = None
        self.tag_language = tag_language.lower()
        self.translator = None
        self.summary_context_window = summary_context_window
        self.summary_llm_model = summary_llm_model or "mistralai/Mistral-7B-Instruct-v0.2"
        if summary_prompt_template:
            self.summary_prompt_template = summary_prompt_template
        else:
            raise ValueError("A summary prompt template must be provided.")

        self.summary_generator = None

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

    def load_summary_generator(self):
        """Load a text generation model for creating summaries and tags."""
        if self.summary_generator is not None:
            return

        print(f"Loading AI summary generator model ({self.summary_llm_model})...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        try:
            # Load the tokenizer and model with 4-bit quantization
            tokenizer = AutoTokenizer.from_pretrained(self.summary_llm_model)
            model = AutoModelForCausalLM.from_pretrained(
                self.summary_llm_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                load_in_4bit=True
            )

            self.summary_generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )
            print("AI summary generator loaded successfully.")
        except Exception as e:
            print(f"Error loading summary generator model: {e}")
            self.summary_generator = None

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

    def _summarize_text_in_chunks(self, text: str) -> str:
        """
        Summarizes long text by splitting it into chunks (Map-Reduce).
        This avoids context window overflow errors.
        """
        if not text:
            return ""

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.summary_llm_model)

        tokens = tokenizer.encode(text)
        max_chunk_size = self.summary_context_window - 50  # Leave room for prompt and model overhead

        if len(tokens) <= max_chunk_size:
            # Text is short enough, no chunking needed
            return text

        print(f"  Audio transcript is too long ({len(tokens)} tokens). Summarizing in chunks...")
        chunks = [tokens[i:i + max_chunk_size] for i in range(0, len(tokens), max_chunk_size)]
        chunk_texts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

        intermediate_summaries = []
        for i, chunk_text in enumerate(chunk_texts):
            print(f"    Summarizing chunk {i+1}/{len(chunk_texts)}...")
            # Simple summarization prompt for each chunk
            prompt = f"Summarize the following text concisely:\n\n{chunk_text}"
            try:
                summary = self.summary_generator(prompt, max_length=150, min_length=20, do_sample=False, truncation=True)[0]['generated_text']
                intermediate_summaries.append(summary)
            except Exception as e:
                print(f"      Warning: Failed to summarize chunk {i+1}. Error: {e}")

        combined_summary = " ".join(intermediate_summaries)
        print(f"  Finished summarizing chunks. Combined length: {len(combined_summary)} chars.")
        return combined_summary



    def generate_summary_and_tags_separately(summary_generator, visual_context, audio_transcript, language):
        """
        Generate summary and tags in two separate calls for better quality
        
        Args:
            summary_generator: The HuggingFace pipeline
            visual_context: Visual descriptions
            audio_transcript: Audio transcript
            language: Target language
            
        Returns:
            Dictionary with 'summary' and 'tags'
        """
        
        # Step 1: Generate summary
        summary_prompt = f"""Summarize this video content in one paragraph in {language}.

    Visual scenes: {visual_context}

    Audio: {audio_transcript}

    Summary:"""
        
        try:
            summary_output = summary_generator(
                summary_prompt,
                max_length=200,
                min_length=30,
                do_sample=False,
                truncation=True
            )[0]['generated_text']
            
            # Clean up output (FLAN-T5 might repeat the prompt)
            summary = summary_output.replace(summary_prompt, "").strip()
            
        except Exception as e:
            print(f"Summary generation failed: {e}")
            summary = ""
        
        # Step 2: Generate tags
        tags_prompt = f"""List 10-15 relevant keywords for this video in {language}, separated by commas.

    Content: {visual_context[:200]}

    Keywords:"""
        
        try:
            tags_output = summary_generator(
                tags_prompt,
                max_length=100,
                do_sample=False,
                truncation=True
            )[0]['generated_text']
            
            # Clean up and parse tags
            tags_text = tags_output.replace(tags_prompt, "").strip()
            tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
            
        except Exception as e:
            print(f"Tag generation failed: {e}")
            tags = []
        
        return {
            'summary': summary,
            'tags': tags
        }

    def parse_llm_output(output: str, fallback_summary: str = "", fallback_tags: list = None) -> dict:
        """
        Robustly parse LLM output even if format is imperfect
        
        Args:
            output: Raw LLM output
            fallback_summary: Use this if parsing fails
            fallback_tags: Use these if parsing fails
            
        Returns:
            Dictionary with 'summary' and 'tags'
        """
        import re
        
        if fallback_tags is None:
            fallback_tags = []
        
        result = {
            'summary': fallback_summary,
            'tags': fallback_tags.copy()
        }
        
        # Try multiple parsing strategies
        
        # Strategy 1: Look for "Summary:" and "Tags:" markers
        summary_match = re.search(r"Summary:\s*(.+?)(?=Tags:|$)", output, re.DOTALL | re.IGNORECASE)
        tags_match = re.search(r"Tags:\s*(.+?)$", output, re.DOTALL | re.IGNORECASE)
        
        if summary_match:
            summary_text = summary_match.group(1).strip()
            # Remove placeholder text
            if '[' not in summary_text and 'here]' not in summary_text.lower():
                result['summary'] = summary_text
        
        if tags_match:
            tags_text = tags_match.group(1).strip()
            # Remove placeholder text
            if '[' not in tags_text and 'here]' not in tags_text.lower():
                # Parse comma-separated tags
                tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
                if tags and tags[0].lower() != 'your':  # Filter out "Your tags here"
                    result['tags'] = tags
        
        # Strategy 2: If no markers found, treat first line as summary, rest as tags
        if not result['summary'] and not result['tags']:
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            if lines:
                result['summary'] = lines[0]
                if len(lines) > 1:
                    # Try to parse remaining lines as tags
                    tags_text = ' '.join(lines[1:])
                    tags = [tag.strip() for tag in re.split(r'[,;]', tags_text) if tag.strip()]
                    result['tags'] = tags[:15]  # Limit to 15 tags
        
        return result


    def generate_ai_summary_and_tags_new(
        self,
        visual_descriptions: list,
        audio_transcript: str,
        language: str
    ) -> dict:
        """
        Generates a consolidated summary and new tags using an LLM.

        Args:
            visual_descriptions: A list of descriptions from the video frames.
            audio_transcript: The transcript from the video's audio.
            language: The target language for the output (e.g., 'German').

        Returns:
            A dictionary with 'summary' and 'tags', or None if generation fails.
        """

        self.load_summary_generator()
        if not self.summary_generator:
            print("Warning: Summary generator not available.")
            return None
        
        # Prepare visual context
        unique_descriptions = sorted(list(set(visual_descriptions)))
        visual_context = "\n".join(f"- {desc}" for desc in unique_descriptions)
        
        # Limit context length to avoid token limits
        if len(visual_context) > 1000:
            visual_context = visual_context[:1000] + "..."
        
        if len(audio_transcript) > 1000:
            audio_transcript = audio_transcript[:1000] + "..."
        
        # Use improved prompt (no placeholders!)
        prompt = f"""Summarize this video in {language}.

    The video shows:
    {visual_context}

    Audio transcript:
    {audio_transcript if audio_transcript else "(no audio)"}

    Task: Write one summary paragraph, then list 10 keywords separated by commas.

    Summary:"""
        
        print(f"\n{'='*60}")
        print("GENERATING AI SUMMARY")
        print(f"{'='*60}")
        print(f"Prompt length: {len(prompt)} characters")
        
        try:
            # Generate with better parameters
            output = self.summary_generator(
                prompt,
                max_length=300,  # Increased for better output
                min_length=50,   # Ensure substantial output
                num_beams=4,
                temperature=0.7,
                do_sample=True,  # Enable sampling for more natural output
                truncation=True,
                early_stopping=True
            )[0]['generated_text']
            
            print(f"\nRaw output ({len(output)} chars):")
            print(output[:500])
            
            # Parse output robustly
            result = self.parse_llm_output(output)
            
            # If parsing failed, try two-step generation
            if not result['summary'] or not result['tags']:
                print("\nFirst attempt failed, trying two-step generation...")
                result = self.generate_summary_and_tags_separately(
                    self.summary_generator,
                    visual_context,
                    audio_transcript,
                    language
                )
            
            if result['summary']:
                print(f"\n✓ Summary: {result['summary'][:100]}...")
            if result['tags']:
                print(f"✓ Tags ({len(result['tags'])}): {', '.join(result['tags'][:5])}...")
            
            print(f"XXXXXXXXXXXXX result: {result}")
            return result if (result['summary'] or result['tags']) else None
            
        except Exception as e:
            print(f"Error during AI summary generation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_ai_summary_and_tags(self, visual_descriptions: List[str], audio_transcript: str, language: str) -> Optional[Dict]:
        """
        Generates a consolidated summary and new tags using an LLM.

        Args:
            visual_descriptions: A list of descriptions from the video frames.
            audio_transcript: The transcript from the video's audio.
            language: The target language for the output (e.g., 'German').

        Returns:
            A dictionary with 'summary' and 'tags', or None if generation fails.
        """
        self.load_summary_generator()
        if not self.summary_generator:
            print("Warning: Summary generator not available. Skipping AI summary.")
            return None

        # Combine unique visual descriptions
        visual_context = ". ".join(sorted(list(set(visual_descriptions))))
        # Combine unique visual descriptions into a cleaner, itemized list
        unique_descriptions = sorted(list(set(visual_descriptions)))
        visual_context = "\n".join(f"- {desc}" for desc in unique_descriptions)

        # Handle potentially long audio transcript using map-reduce summarization
        # The summarizer model itself has a max length (e.g., 512 for t5-base)
        summarized_audio_transcript = self._summarize_text_in_chunks(audio_transcript)

        # Build the prompt
        # Build the prompt - a more direct, few-shot prompt can improve compliance
        prompt = self.summary_prompt_template.format(
            visual_context=visual_context,
            audio_transcript=summarized_audio_transcript,
            language=language
        )

        print("xxxxxxxxxxxxxxxx PROMPT:", prompt)

        try:
            print("  Generating AI summary and tags...")
            output = self.summary_generator(
                prompt,
                max_new_tokens=300,  # Max tokens to generate
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                pad_token_id=self.summary_generator.tokenizer.eos_token_id,
                eos_token_id=self.summary_generator.tokenizer.eos_token_id,
                num_return_sequences=1,
                truncation=True
            )[0]['generated_text']

            print("xxxxxxxxxxxxxxxx OUTPUT:", output)

            # Parse the output
            summary_match = re.search(r"Summary:\s*(.*?)Tags:", output, re.DOTALL | re.IGNORECASE)
            tags_match = re.search(r"Tags:\s*(.*)", output, re.IGNORECASE)

            summary = summary_match.group(1).strip() if summary_match else ""
            tags_str = tags_match.group(1).strip() if tags_match else ""
            tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]

            if len(summary) == 0 and len(tags) == 0:
                print("  Warning: AI summary generation returned empty results.")
                return None
            
            print(f"  AI Summary generated ({len(summary)} chars).")
            print(f"  AI Tags generated: {len(tags)} tags.")

            return {'summary': summary, 'tags': tags}
        except Exception as e:
            print(f"Error during AI summary generation: {e}")
            return None

    def cleanup(self):
        """Free up GPU memory"""
        del self.model
        del self.processor
        if self.translator:
            del self.translator
        self.model = None
        self.processor = None
        self.translator = None
        if self.summary_generator:
            del self.summary_generator
            self.summary_generator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("AI analyzer cleaned up.")