# Sports Rules Companion: AI Assistant for Sports Officials

A proof-of-concept Q&A system exploring how sports officials can learn and apply rulebook knowledge using AI. Built with Google Cloud's Vertex AI technology, it demonstrates natural language interaction with our comprehensive [Rulebooks](./Rulebooks/) collection - featuring official PDFs from MLB, NBA, NFL, NHL, Cricket, Golf, Soccer, and Skiing. Whether you're studying to become an official or exploring rule interpretations, this project shows how AI can help navigate complex sports regulations.

## Playground Instructions

### Prerequisites

1. **Google Cloud Account**: Ensure you have a Google Cloud account.
2. **Google Cloud SDK**: Install the Google Cloud SDK on your local machine. Follow the instructions [here](https://cloud.google.com/sdk/docs/install).

### Setup

1. **Clone the Repository**:

    ```sh
    git clone https://github.com/dzivkovi/sports-rules-companion.git
    cd sports-rules-companion
    ```

2. **Install Dependencies**:

    ```sh
    pip install -r requirements.txt
    ```

3. **Authenticate with Google Cloud**:

    ```sh
    gcloud auth login
    gcloud config set project YOUR_PROJECT_ID
    ```

4. **Configure Git LFS**:

    ```sh
    git lfs install
    git lfs pull
    ```

    This step ensures proper handling of large rulebook PDFs.

### Running Notebooks

After setup, you can explore the Q&A capabilities through our Jupyter notebooks that demonstrate different approaches for each sport:

1. **Start Jupyter**:

    ```sh
    jupyter notebook
    ```

2. **Open a Notebook**: Navigate to the notebook you are interested in and open it.

## Example Workflows

Jupyter notebooks demonstrate how to build Q&A systems over sports rulebooks using Google Cloud's AI services. Each sport's implementation showcases different aspects of document processing and natural language understanding:

- Baseball (MLB) - Using Document AI and Vertex AI for text parsing and Q&A
- Golf (USGA) - Vector embeddings for efficient rule search
- Soccer (IFAB) - Context-aware responses with section references
- Cricket (MCC) - Intelligent rule interpretation
- Basketball (NBA) - Document parsing and natural language Q&A
- Football (NFL) - Semantic search across rulebook sections
- Hockey (NHL) - Automated rule lookup and explanation
- Skiing (FIS) - Technical regulations and gate judging rules

Each notebook demonstrates a consistent workflow:

1. Parse PDF rulebooks using Document AI
2. Create searchable embeddings using Vertex AI
3. Generate contextual answers using LLMs
4. Reference specific rule sections

For implementation details and the underlying architecture, see [Readme-legacy.md](./Readme-legacy.md).

## Credits & License

This project builds upon Mike Henderson's fantastic [vertex-ai-mlops](https://github.com/statmike/vertex-ai-mlops) repository, which provides comprehensive examples of Google Cloud's Vertex AI capabilities. His educational examples in the "Applied GenAI" folder, particularly the document Q&A system, provided the foundation for extending these capabilities to sports officiating use cases.

The original repository and this project are licensed under the Apache License 2.0. See Mike's [LICENSE](https://github.com/statmike/vertex-ai-mlops/blob/main/LICENSE) for details.
