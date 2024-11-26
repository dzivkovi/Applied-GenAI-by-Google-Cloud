#!/usr/bin/env python
"""
Sport Rules Q&A using Google Gemini 1.5 with support for various storage backends.

Examples:
    # Default to Soccer and FIFA rules
    python sport_rules_qa.py

    # Local file
    python sport_rules_qa.py --input ./rules.pdf

    # HTTP(S)
    python sport_rules_qa.py --input https://example.com/rules.pdf

    # Amazon S3
    python sport_rules_qa.py --input s3://bucket/rules.pdf

    # Google Cloud Storage
    python sport_rules_qa.py --input gs://bucket/rules.pdf
"""

import os
import logging
import tempfile
import argparse
import requests
import fsspec
import s3fs
import gcsfs
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DEFAULT_MODEL = "gemini-1.5-flash"
DEFAULT_PDF_URL = "https://digitalhub.fifa.com/m/5371a6dcc42fbb44/original/Law-Book-2023-24-English.pdf"

# Configure the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable must be set!")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Constants
DEFAULT_MODEL = "gemini-1.5-flash"
DEFAULT_SPORT = "Soccer/Football"
DEFAULT_PDF_URL = "https://digitalhub.fifa.com/m/5371a6dcc42fbb44/original/Law-Book-2023-24-English.pdf"


class CloudStorageHandler:
    """Handles file access across different storage systems."""
    @staticmethod
    def read_file(file_path: str) -> bytes:
        """Read file from any supported storage system."""
        try:
            content = None

            # Handle HTTP(S) URLs directly with requests
            if file_path.startswith(("http://", "https://")):
                response = requests.get(file_path, timeout=60)
                response.raise_for_status()
                content = response.content

            # Handle local files with direct path
            elif os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    content = f.read()

            # Handle cloud storage (S3, GCS) with fsspec
            else:
                protocol = file_path.split("://")[0] if "://" in file_path else "file"
                if protocol == "s3":
                    # Fix S3 configuration
                    s3 = s3fs.S3FileSystem(
                        anon=False,  # Use credentials
                        key=os.getenv("AWS_ACCESS_KEY_ID"),
                        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
                        client_kwargs={
                            "region_name": os.getenv("AWS_REGION", "us-east-1")
                        }
                    )
                    with s3.open(file_path, "rb") as f:
                        content = f.read()
                elif protocol == "gs":
                    fs = gcsfs.GCSFileSystem(
                        token=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                    )
                    with fs.open(file_path, "rb") as f:
                        content = f.read()
                else:
                    with fsspec.open(file_path, "rb") as f:
                        content = f.read()

            if content:
                return content
            else:
                raise ValueError("No content read from file")

        except Exception as e:
            msg = f"Error reading file from {file_path}: {e}"
            raise Exception(msg)


class SportRulesQA:
    """Main QA system using Gemini."""

    SYSTEM_PROMPT_TEMPLATE = """
    You are an expert on {sport_name} rules and regulations. Your task is to:
    1. Provide accurate answers based on the official rules document provided
    2. Always cite the specific law or section number when providing answers
    3. Be concise but comprehensive
    4. If a question cannot be answered using the provided document, say so clearly

    Remember to:
    - Focus only on official rules, not interpretations or opinions
    - Quote relevant text when appropriate
    - Use bullet points for multiple rules or steps
    """

    def __init__(self, sport_name: str = DEFAULT_SPORT, model_name: str = DEFAULT_MODEL):
        """Initialize the QA system with specified Gemini model."""
        self.sport = sport_name
        self.model = genai.GenerativeModel(model_name)
        # Format the system prompt with the sport name
        self.system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(sport_name=self.sport)
        self.document = None
        logging.info("Initialized with model: %s", model_name)

    def load_document(self, file_path: str) -> None:
        """Load document from any supported storage location."""
        try:
            # Read document content
            content = CloudStorageHandler.read_file(file_path)

            # Save to temporary file for Gemini upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(content)
                temp_file.flush()
                self.document = genai.upload_file(temp_file.name)
                print(f"Document loaded successfully from '{file_path}'")

        except Exception as e:
            msg = f"Error loading document: {e}"
            raise Exception(msg)
        finally:
            if 'temp_file' in locals():
                os.unlink(temp_file.name)

    def ask(self, question: str) -> str:
        """Ask a question about the sport rules."""
        if not self.document:
            return "Error: No document loaded. Please load a document first."

        try:
            response = self.model.generate_content(
                [self.document, self.system_prompt, f"Question: {question}\nAnswer:"],
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                ),
            )
            return response.text
        except Exception as e:
            return f"Error generating answer: {e}"


def get_separator(text: str) -> str:
    """Create separator line matching text length."""
    return "-" * len(text)


def main():
    """Main entry point for the sport rules QA system."""
    parser = argparse.ArgumentParser(
        description="Sport Rules Q&A System using Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_PDF_URL,
        help="Input PDF document path (local, S3, GCS, or HTTP(S))",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument("--sport", default=DEFAULT_SPORT, help="Sport name")
    args = parser.parse_args()

    try:
        qa = SportRulesQA(sport_name=args.sport, model_name=args.model)
        print(f"Loading Sport Rules Q&A System for '{args.sport}'")
        qa.load_document(args.input)

        # Initial document validation and identification
        test = qa.ask(
            "Please identify what official rulebook or document this is and what sport it covers. Keep it brief."
        )
        if "400" in test:
            print(f"\nError: {test}")  # Print full error message
            return 1
        print(f"Document description: {test}")

        prompt = f"\nAsk a question about {args.sport} rules (type 'quit' to exit)"
        print(prompt)
        print(get_separator(prompt))

        while True:
            question = input("\nAsk a question: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                break
            print(f"\nAnswer: {qa.ask(question)}")

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")  # Print full error message
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
