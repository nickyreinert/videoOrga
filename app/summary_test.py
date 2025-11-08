"""
Temporary Test Script for AI Summary Generation

This script is for debugging and refining the AI summary and tag generation logic.
It uses static data to simulate the inputs from the main application, allowing for
isolated testing of the language model's output and parsing functions.

Run this file directly to test the summarization:
python summary_test.py
"""

import torch
import re
from typing import List, Dict, Optional

# --- Static Example Data ---

# A good prompt gives clear instructions to the model.
# It defines the context, the task, and the desired output format.
# This prompt is more direct and less likely to cause the model to repeat the input.
PROMPT = """Write a very brief, one-paragraph summary in {language} for a video with the following content. The summary should combine both the visual and audio information. After the summary, provide a comma-separated list of 10 relevant keywords.

Visuals: A person is walking on a sandy beach next to the ocean.
Audio: A person is enjoying a peaceful walk on the beach, listening to the waves.

Summary in English: A person enjoys a peaceful walk on a beautiful beach, with the sound of the waves creating a calm atmosphere.
Tags in English: beach, walking, sand, ocean, waves, peaceful, relaxing, nature, outdoor, vacation

---

Visuals: {visual_context}
Audio: {audio_transcript}

Summary in {language}:
Tags in {language}:"""

VISUAL_DESCRIPTIONS = [
    "a drone shot of a person walking on a beach",
    "a close up of a person walking on the sand",
    "a person walking on a beach with waves in the background",
    "a wide shot of a beach with a person walking towards the water"
]

AUDIO_TRANSCRIPT = (
    "I'm walking along the shore today, and the weather is just perfect. "
    "The sun is out, but it's not too hot. You can hear the waves crashing, "
    "it's so peaceful. I think I'll just keep walking for a while and enjoy this moment. "
    "It's a great day to be at the beach, feeling the sand between my toes."
)

LANGUAGE = "German"

# --- Core Logic from AIAnalyzer ---

class SummaryTester:
    """
    A simplified class containing only the summary generation logic
    for focused testing.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.summary_llm_model = "google/flan-t5-large"
        self.summary_generator = None
        print(f"Using device: {self.device}")

    def load_summary_generator(self):
        """Load a text generation model for creating summaries and tags."""
        if self.summary_generator is not None:
            return

        print(f"Loading AI summary generator model ({self.summary_llm_model})...")
        from transformers import pipeline

        try:
            self.summary_generator = pipeline(
                "text2text-generation",
                model=self.summary_llm_model,
                device=0 if self.device == "cuda" else -1
            )
            print("AI summary generator loaded successfully.")
        except Exception as e:
            print(f"Error loading summary generator model: {e}")
            self.summary_generator = None

    def parse_llm_output(self, output: str) -> dict:
        """
        Robustly parse LLM output to extract summary and tags.
        """
        summary = ""
        tags = []

        # The model might output "Summary:" and "Tags:" markers.
        # If not, we can assume the first part is the summary and the second is the tags.
        if "Tags:" in output:
            summary_part, tags_part = output.split("Tags:", 1)
            summary = summary_part.replace("Summary:", "").strip()
            tags_text = tags_part.strip()
            tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
        else:
            # Fallback if "Tags:" marker is missing
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            if lines:
                summary = lines[0]
                if len(lines) > 1:
                    tags_text = ' '.join(lines[1:])
                    tags = [tag.strip() for tag in re.split(r'[,;]', tags_text) if tag.strip()]

        # Final fallback if parsing fails completely
        if not summary and not tags and output:
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            if lines:
                # Assume first non-empty line is the summary
                summary = lines[0]
                # Assume the rest contains tags
                if len(lines) > 1:
                    tags_text = ' '.join(lines[1:])
                    tags = [tag.strip() for tag in re.split(r'[,;]', tags_text) if tag.strip()]

        return {'summary': summary, 'tags': tags}

    def generate_ai_summary_and_tags(
        self,
        visual_descriptions: list,
        audio_transcript: str,
        language: str
    ) -> dict:
        """
        Generates a consolidated summary and new tags using an LLM.
        """
        self.load_summary_generator()
        if not self.summary_generator:
            print("Warning: Summary generator not available.")
            return None

        # Prepare context
        visual_context = ", ".join(sorted(list(set(visual_descriptions))))

        prompt = PROMPT.format(
            language=language,
            visual_context=visual_context,
            audio_transcript=audio_transcript
        )

        print("\n" + "="*80)
        print("PROMPT SENT TO MODEL:")
        print(prompt)
        print("="*80 + "\n")

        try:
            output = self.summary_generator(
                prompt,
                max_length=300,
                min_length=30,
                num_beams=4,
                temperature=0.8,   # Increase temperature to allow more creativity
                do_sample=False,    # Enable sampling to avoid repetitive output
                truncation=False,
                early_stopping=False
            )[0]['generated_text']

            print("\n" + "="*80)
            print("RAW MODEL OUTPUT:")
            print(output)
            print("="*80 + "\n")

            # Parse the output
            result = self.parse_llm_output(output)
            return result

        except Exception as e:
            print(f"Error during AI summary generation: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    tester = SummaryTester()
    
    # Run the generation and parsing logic
    final_result = tester.generate_ai_summary_and_tags(
        visual_descriptions=VISUAL_DESCRIPTIONS,
        audio_transcript=AUDIO_TRANSCRIPT,
        language=LANGUAGE
    )

    print("\n" + "="*80)
    print("PARSED RESULT:")
    if final_result:
        print(f"\nSummary: {final_result.get('summary', 'N/A')}")
        print(f"\nTags: {final_result.get('tags', 'N/A')}")
    else:
        print("Failed to generate or parse result.")
    print("="*80 + "\n")
