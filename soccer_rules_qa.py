import argparse
import io
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
    DEFAULT_RULES_URL = "https://digitalhub.fifa.com/m/5371a6dcc42fbb44/original/Law-Book-2023-24-English.pdf"
    
    def __init__(self, 
                 project_id: str,
                 region: str = "us-central1",
                 source_url: Optional[str] = None,
                 cache_dir: Optional[str] = "cache"):
        """Initialize the Document QA system."""
        self.project_id = project_id
        self.region = region
        self.source_url = source_url or self.DEFAULT_RULES_URL
        self.cache_dir = Path(cache_dir)
        
        try:
            # Create cache directory
            self.cache_dir.mkdir(exist_ok=True)
            
            # Initialize Vertex AI
            vertexai.init(project=project_id, location=region)
            
            # Initialize models
            self.embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
            self.llm = TextGenerationModel.from_pretrained("text-bison@002")
            
            # Process document
            self._process_document()
            
        except Exception as e:
            raise Exception(f"Failed to initialize DocumentQA: {str(e)}")

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
                text = page.extract_text()
                embedding = []
                if text:
                    embedding = self.embedding_model.get_embeddings([text])[0].values
    
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
    
                self.page_images.append(self._create_page_image(page))
    
            # Build FAISS index
            valid_embeddings = [doc["embedding"] for doc in self.documents if len(doc["embedding"]) > 0]
            if valid_embeddings:
                embeddings = np.array(valid_embeddings).astype("float32")
                self.faiss_index = self._build_faiss_index(embeddings)
            else:
                raise ValueError("No valid embeddings found in document")
    
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build a FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index
        
    def _get_page_vertices(self, page) -> List[Dict]:
        """Extract page vertices."""
        # Implementation depends on PDF structure
        return []
        
    def _create_page_image(self, page) -> PIL.Image:
        """Create image from PDF page."""
        # Implementation depends on PDF structure
        return PIL.Image.new('RGB', (100, 100))

    def ask(self, question: str, k: int = 3) -> str:
        """Answer a question about the document."""
        try:
            # Get question embedding
            question_embedding = self.embedding_model.get_embeddings([question])[0].values
            
            # Search similar passages
            D, I = self.faiss_index.search(
                np.array([question_embedding]).astype("float32"), k
            )
            
            # Build context
            context = "\n".join([
                f"Page {self.documents[i]['metadata']['page']}: {self.documents[i]['page_content']}"
                for i in I[0]
            ])
            
            # Generate answer
            prompt = f"""Based on the following context from the soccer rules document, 
            answer this question: {question}\n\nContext:\n{context}"""
            
            response = self.llm.predict(prompt, temperature=0.2)
            return response.text
            
        except Exception as e:
            return f"Error answering question: {str(e)}"

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Soccer Rules QA System")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--pdf-url", default=None, help="URL to PDF document")
    args = parser.parse_args()
    
    try:
        qa = DocumentQA(
            project_id=args.project_id,
            source_url=args.pdf_url
        )
        
        while True:
            question = input("\nAsk a question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            answer = qa.ask(question)
            print(f"\nAnswer: {answer}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
