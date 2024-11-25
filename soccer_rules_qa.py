#!/usr/bin/env python
"""
Soccer Rules Q&A using Google's Gemini Pro 1.5.
Leverages Gemini's large context window to process entire PDF documents without splitting.

Example usage:
    python soccer_rules_qa.py --input https://digitalhub.fifa.com/m/5371a6dcc42fbb44/original/Law-Book-2023-24-English.pdf
"""

import os
import sys
import logging
from typing import Optional
import argparse
import tempfile
import requests
import fsspec
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)

# Configure the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable must be set!")
genai.configure(api_key=GOOGLE_API_KEY)

# Constants
DEFAULT_MODEL = "gemini-1.5-flash"
DEFAULT_PDF_URL = "https://digitalhub.fifa.com/m/5371a6dcc42fbb44/original/Law-Book-2023-24-English.pdf"

SYSTEM_PROMPT = """
You are an expert on soccer/football rules and regulations. Your task is to:
1. Provide accurate answers based on the official rules document provided
2. Always cite the specific law or section number when providing answers
3. Be concise but comprehensive
4. If a question cannot be answered using the provided document, say so clearly

Remember to:
- Focus only on official rules, not interpretations or opinions
- Quote relevant text when appropriate
- Use bullet points for multiple rules or steps
"""


class SoccerRulesQA:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize the QA system with specified Gemini model."""
        self.model = genai.GenerativeModel(model_name)
        self.document = None
        logging.info(f"Initialized with model: {model_name}")

    def load_document(self, file_path: str) -> None:
        """Load document from URL or local path."""
        try:
            # Read document content
            if file_path.startswith(("http://", "https://")):
                response = requests.get(file_path, timeout=60)
                response.raise_for_status()
                content = response.content
            else:
                with fsspec.open(file_path, "rb") as file:
                    content = file.read()

            # Save to temporary file for Gemini upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(content)
                self.document = genai.upload_file(temp_file.name)
                logging.info(f"Document loaded successfully from {file_path}")

        except Exception as e:
            raise Exception(f"Error loading document: {str(e)}")
        finally:
            if "temp_file" in locals():
                os.unlink(temp_file.name)

    def ask(self, question: str) -> str:
        """Ask a question about the soccer rules."""
        if not self.document:
            return "Error: No document loaded. Please load a document first."

        try:
            # Combine document, system prompt, and question
            response = self.model.generate_content(
                [self.document, SYSTEM_PROMPT, f"Question: {question}\nAnswer:"],
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                ),
            )

            return response.text

        except Exception as e:
            return f"Error generating answer: {str(e)}"


def main():
    """Main entry point for the soccer rules QA system."""
    parser = argparse.ArgumentParser(description="Soccer Rules Q&A System using Gemini")
    parser.add_argument(
        "--input",
        default=DEFAULT_PDF_URL,
        help="Input PDF document path (local, S3, GCS, or HTTP(S))",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    args = parser.parse_args()

    try:
        # Initialize QA system
        qa = SoccerRulesQA(model_name=args.model)

        # Load document
        print(f"Loading document from {args.input}...")
        qa.load_document(args.input)

        # Interactive Q&A loop
        print("\nSoccer Rules Q&A System (type 'quit' to exit)")
        print("-" * 50)

        while True:
            question = input("\nAsk a question: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                break

            answer = qa.ask(question)
            print(f"\nAnswer: {answer}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
