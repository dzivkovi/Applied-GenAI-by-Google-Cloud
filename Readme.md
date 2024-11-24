# Applied Generative AI - Sports Rulebooks Q&A

Welcome to the Applied Generative AI repository focused on Q&A with sports rulebooks. This repository contains various Sports Rulebooks in the `[Rulebooks](./Rulebooks)` folder and provides tools and examples for using Vertex AI to create Q&A systems based on these documents.

It is a slightly modified subfolder of Mike Henderson's fantastic [Vertex AI MLOps repo](https://github.com/statmike/vertex-ai-mlops). Refer to [Readme-legacy.md](./Readme-legacy.md) file for more details about files copied from the "Applied GenAI" sub-folder.

## Playground Instructions

Welcome to my playground for exploring Generative AI with Google Cloud! Here are some instructions to get you started:

### Prerequisites

1. **Google Cloud Account**: Ensure you have a Google Cloud account.
2. **Google Cloud SDK**: Install the Google Cloud SDK on your local machine. Follow the instructions [here](https://cloud.google.com/sdk/docs/install).

### Setup

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/dzivkovi/Applied-GenAI-by-Google-Cloud.git
    cd Applied-GenAI-by-Google-Cloud
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

### Running Notebooks

1. **Jupyter Notebook**: Start Jupyter Notebook to explore the notebooks.
    ```sh
    jupyter notebook
    ```

2. **Open a Notebook**: Navigate to the notebook you are interested in and open it.

### Using Git Large File Storage (LFS)

This repository uses Git Large File Storage (LFS) to manage large files efficiently. If you are cloning this repository or working with large files, please follow these steps:

1. **Install Git LFS**:
    ```sh
    git lfs install
    ```

2. **Clone the Repository**:
    ```sh
    git clone https://github.com/dzivkovi/Applied-GenAI-by-Google-Cloud.git
    cd Applied-GenAI-by-Google-Cloud
    ```

3. **Pull LFS Files**:
    ```sh
    git lfs pull
    ```

By following these steps, you ensure that large files are handled properly and won't cause issues for people checking out your code.

## Example Workflows

This repository covers Q&A systems for various sports rulebooks, including:
- Baseball (MLB)
- Golf (USGA)
- Soccer (IFAB)
- Cricket (MCC)
- Basketball (NBA)
- Football (NFL)
- Hockey (NHL)

For more detailed examples and workflows, refer to the [Readme-legacy.md](./Readme-legacy.md) file.