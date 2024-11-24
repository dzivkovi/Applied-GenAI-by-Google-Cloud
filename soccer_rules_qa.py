import os
import io
import json
import base64
import argparse
import requests
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import faiss
import PyPDF2
import PIL
import PIL.ImageFont, PIL.Image, PIL.ImageDraw
import shapely

from google.cloud import aiplatform
from google.cloud import documentai
from google.cloud import storage
import vertexai
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel


class DocumentQA:
    def __init__(self, 
                 project_id: str,
                 region: str = "us-central1",
                 source_url: Optional[str] = None,
                 cache_dir: Optional[str] = "cache"):
        """
        Initialize the Document QA system.
        
        Args:
            project_id: Google Cloud project ID
            region: Google Cloud region
            source_url: URL to the PDF document
            cache_dir: Directory to cache processed documents and embeddings
        """
        self.project_id = project_id
        self.region = region
        self.source_url = source_url or "https://www.theifab.com/laws-of-the-game-documents/?language=all&year=2022%2F23"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize Google Cloud clients
        vertexai.init(project=project_id, location=region)
        self.embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
        self.generation_model = TextGenerationModel.from_pretrained("text-bison@002")

        # Initialize document processing components
        self.documents = []
        self.page_images = []
        self.faiss_index = None

        # Load or process documents
        if self._cache_exists():
            self._load_from_cache()
        else:
            self._process_document()
            self._save_to_cache()

    def _cache_exists(self) -> bool:
        """Check if cached files exist."""
        cache_files = ["documents.json", "embeddings.npy", "images.pkl"]
        return all((self.cache_dir / f).exists() for f in cache_files)

    def _load_from_cache(self):
        """Load processed documents from cache."""
        with open(self.cache_dir / "documents.json", "r") as f:
            self.documents = json.load(f)

        embeddings = np.load(self.cache_dir / "embeddings.npy")
        self.faiss_index = self._build_faiss_index(embeddings)

        import pickle
        with open(self.cache_dir / "images.pkl", "rb") as f:
            self.page_images = pickle.load(f)

    def _save_to_cache(self):
        """Save processed documents to cache."""
        with open(self.cache_dir / "documents.json", "w") as f:
            json.dump(self.documents, f)

        embeddings = np.array([doc["embedding"] for doc in self.documents]).astype("float32")
        np.save(self.cache_dir / "embeddings.npy", embeddings)

        import pickle
        with open(self.cache_dir / "images.pkl", "wb") as f:
            pickle.dump(self.page_images, f)

    def _process_document(self):
        """Process the source document and create embeddings."""
        try:
            # Download PDF
            response = requests.get(self.source_url)
            response.raise_for_status()
            pdf = PyPDF2.PdfReader(io.BytesIO(response.content))
    
            self.documents = []
            self.page_images = []
    
            # Process each page
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text and create image
                text = page.extract_text()
    
                # Create embedding
                embedding = []
                if text:
                    embedding = self.embedding_model.get_embeddings([text])[0].values
    
                # Store document info
                self.documents.append({
                    "page_content": text,
                    "metadata": {
                        "page": page_num,
                        "source_document": self.source_url,
                        "filename": self.source_url.split("/")[-1],
                        "vme_id": str(page_num - 1)
                    },
                    "embedding": embedding,
                    "extras": {
                        "vertices": self._get_page_vertices(page)
                    }
                })
    
                # Store page image
                self.page_images.append(self._create_page_image(page))
    
            # Build FAISS index from non-empty embeddings
            valid_embeddings = [doc["embedding"] for doc in self.documents if len(doc["embedding"]) > 0]
            if valid_embeddings:
                embeddings = np.array(valid_embeddings).astype("float32")
                self.faiss_index = self._build_faiss_index(embeddings)
            else:
                raise ValueError("No valid embeddings found in document")
    
        except requests.RequestException as e:
            raise Exception(f"Failed to download PDF: {e}")
        except PyPDF2.PdfReadError as e:
            raise Exception(f"Failed to read PDF: {e}")
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build a FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def _get_page_vertices(self, page) -> List[Dict[str, float]]:
        """Get page vertices for highlighting."""
        # Simplified version - returns corners of the page
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        return [
            {"x": 0, "y": 0},
            {"x": width, "y": 0},
            {"x": width, "y": height},
            {"x": 0, "y": height}
        ]

    def _create_page_image(self, page) -> PIL.Image:
        """Create an image of the page."""
        # Simplified version - creates a blank image
        width = int(page.mediabox.width)
        height = int(page.mediabox.height)
        return PIL.Image.new("RGB", (width, height), "white")

    def answer_question(self, question: str, max_output_tokens: int = 300) -> Dict[str, Any]:
        """
        Answer a question about the document.
        
        Args:
            question: The question to answer
            max_output_tokens: Maximum number of tokens in the response
            
        Returns:
            Dictionary containing the answer and related information
        """
        # Get query embedding and search
        query_vector = np.array([self.embedding_model.get_embeddings([question])[0].values]).astype("float32")
        faiss.normalize_L2(query_vector)
        distances, neighbors = self.faiss_index.search(query_vector, k=3)

        # Prepare context
        context = "\n".join(
            [
                f"Context {i+1}:\n{self.documents[idx]['page_content']}"
                for i, idx in enumerate(neighbors[0])
            ]
        )

        # Generate answer
        prompt = f"""
        Give a detailed answer to the question using information from the provided contexts.
        
        {context}
        
        Question: {question}
        
        Answer and Explanation:
        """

        response = self.textgen_model.predict(prompt, max_output_tokens=max_output_tokens)

        return {
            "question": question,
            "answer": response.text,
            "contexts": [self.documents[idx] for idx in neighbors[0]],
            "scores": distances[0].tolist()
        }

def main():
    parser = argparse.ArgumentParser(description="Soccer Rules Q&A System")
    parser.add_argument("--project-id", required=True, help="Google Cloud Project ID")
    parser.add_argument("--region", default="us-central1", help="Google Cloud Region")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory")
    args = parser.parse_args()

    # Initialize the QA system
    qa = DocumentQA(
        project_id=args.project_id,
        region=args.region,
        cache_dir=args.cache_dir
    )

    # Interactive Q&A loop
    print("Soccer Rules Q&A System (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() in ["quit", "exit", "q"]:
            break
            
        result = qa.answer_question(question)
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for ctx, score in zip(result["contexts"], result["scores"]):
            print(f"- Page {ctx['metadata']['page']}, relevance: {score:.2f}")


if __name__ == "__main__":
    main()
