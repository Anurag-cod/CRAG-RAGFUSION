
# AI-Enhanced Document Search Optimization with Corrective and Fusion RAG Techniques, Deployed via Jenkins and AWS

## Overview

Robust document processing and query generation system that integrates advanced language models with Corrective RAG search and RAG Fusion techniques. The application leverages OpenAI's language models to refine and generate responses based on user queries and documents.

## Project Structure

```
CRAG+RAGFUSION/
│
├── src/
│   ├── __init__.py             # Treats the folder as a package
│   ├── pdf_processing.py       # Handles PDF document processing
│   ├── query_generation.py      # Generates search queries from user input
│   ├── web_search.py            # Performs web searches for additional information
│   ├── llm_integration.py       # Integrates with the LLM (Large Language Model) for various tasks
│   ├── document_fusion.py       # Fuses multiple documents into a coherent set of information
│   ├── query_transformation.py  # Refines and transforms user queries
│   └── response_generation.py   # Generates responses based on the processed queries and documents
│
└── streamlit_app.py             # Main application file for user interaction using Streamlit
```

### Modular Components

- **`pdf_processing.py`**: Extracts and processes text from PDF documents.
- **`query_generation.py`**: Generates search queries based on user input.
- **`web_search.py`**: Performs web searches to gather additional information if needed.
- **`llm_integration.py`**: Handles interactions with the Large Language Model (LLM).
- **`document_fusion.py`**: Fuses multiple documents into a cohesive set of information.
- **`query_transformation.py`**: Refines and transforms user queries to enhance search effectiveness.
- **`response_generation.py`**: Generates responses based on the processed queries and documents.
- **`streamlit_app.py`**: Provides the user interface using Streamlit.

## Setup and Installation

### Prerequisites

- Python 3.x
- Docker
- AWS CLI (if deploying to AWS)

### Local Setup

1. **Clone the Repository**

    ```bash
    git clone <repository-url>
    cd CRAG+RAGFUSION
    ```

2. **Install Dependencies**

    It is recommended to use a virtual environment. Install required packages using:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run Streamlit App Locally**

    ```bash
    streamlit run streamlit_app.py
    ```

### Docker Setup

1. **Build Docker Image**

    ```bash
    docker build -t your-image-name .
    ```

2. **Run Docker Container**

    ```bash
    docker run -p 8501:8501 your-image-name
    ```

### Deployment

**AWS Deployment** involves the following steps:

1. **Create EC2 Instances**

   - **Instance 1**: Jenkins CI/CD server for automating build and deployment processes.
   - **Instance 2**: Application deployment server running Docker containers.

2. **Setup Jenkins**

   - Install Jenkins on Instance 1.
   - Configure Jenkins pipelines for continuous integration and deployment.

3. **Configure Docker and AWS CLI**

   - Install Docker and AWS CLI on Instance 2.
   - Configure Docker to pull images from AWS ECR.
   - Deploy the application using Docker on Instance 2.

4. **Elastic IP Configuration**

   - Assign an Elastic IP to Instance 2 for stable access.

### Jenkins Pipeline

The Jenkins pipeline automates the build and deployment processes:

- **Continuous Integration**: Linting and testing of the codebase.
- **Login to ECR**: Authenticates Docker to push images to AWS ECR.
- **Build and Push Docker Image**: Builds the Docker image and pushes it to ECR.
- **Continuous Deployment**: Deploys the Docker image to the application server using `docker-compose`.

### Docker Compose

The `docker-compose.yml` file manages multi-container Docker applications, allowing for easy configuration and deployment of your application stack.

### Additional Files

- **`setup.py`**: Not required if using Docker; used for installing dependencies in a Python environment.
- **`config.yml`**: Optional configuration file for specific deployment or build settings.

---

This README provides a detailed overview of the project, setup instructions, and deployment details. It should give users a clear understanding of the project's structure and how to get it up and running. If you have any more questions or need further adjustments, let me know!
