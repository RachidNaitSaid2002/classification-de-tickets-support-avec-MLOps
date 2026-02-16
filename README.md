# IT Support Ticket Classification with MLOps

An end-to-end machine learning pipeline for automatically classifying IT support tickets using NLP and modern MLOps practices.

## Overview

This project implements a complete batch NLP pipeline that:
- Processes raw IT support tickets
- Generates multilingual text embeddings using sentence transformers
- Stores embeddings in ChromaDB vector database
- Trains a classification model using scikit-learn
- Monitors data drift using Evidently
- Deploys via Docker and Kubernetes
- CI/CD automation with GitHub Actions

## Project Structure

```
.
├── data/                       # Data directories
│   ├── raw/                    # Raw data (dataset.csv)
│   └── processed/             # Cleaned data
├── src/                        # Source code
│   ├── load_data.py           # Load data from ChromaDB
│   ├── embeddings.py          # Text embedding generation
│   ├── save_vectors.py        # Vector storage management
│   ├── save_chromadb.py       # Save embeddings to ChromaDB
│   └── model_training.py      # Model training pipeline
├── monitoring/                 # Monitoring & drift detection
│   └── drift_detection.py     # Data drift analysis with Evidently
├── models/                     # Trained models
├── chroma/                     # ChromaDB vector store
├── k8s/                        # Kubernetes manifests
│   └── deployment.yaml        # K8s deployment configuration
├── .github/workflows/          # CI/CD pipelines
│   └── main.yml               # GitHub Actions workflow
├── docker-compose.yml          # Docker Compose for monitoring
├── Dockerfile                  # Application container
├── requirements.txt            # Python dependencies
└── main.py                     # Entry point
```

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.12 |
| ML Framework | scikit-learn |
| Embeddings | Sentence Transformers (multilingual-e5-base) |
| Vector Store | ChromaDB |
| Monitoring | Evidently, Prometheus, Grafana |
| Container | Docker, Docker Compose |
| Orchestration | Kubernetes |
| CI/CD | GitHub Actions |

## Installation

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Kubernetes (optional)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd classification-de-tickets-support-avec-MLOps

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install flake8 ruff black pytest
```

## Usage

### 1. Data Processing & Embeddings

Generate embeddings from raw text data:

```python
from src.embeddings import EmbeddingGenerator
from src.save_chromadb import process_and_store

process_and_store()
```

### 2. Model Training

Train the classification model:

```python
python -m src.model_training
```

Or via Docker:

```bash
docker build -t ticket-classifier .
docker run ticket-classifier
```

### 3. Run Inference

```python
from src.load_data import load_data_from_chroma
from src.model_training import train_model
import joblib

# Load data
Features, labels = load_data_from_chroma()

# Load trained model
model = joblib.load("models/model.joblib")
encoder = joblib.load("models/encoder.joblib")

# Predict
predictions = model.predict(Features)
predicted_labels = encoder.inverse_transform(predictions)
```

### 4. Drift Detection

Monitor data drift:

```python
python -m monitoring.drift_detection
```

View the generated HTML report at `monitoring/drift_report.html`.

## Monitoring Stack

Run the monitoring infrastructure:

```bash
docker-compose up -d
```

Services:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (default credentials: admin/admin)
- **cAdvisor**: http://localhost:8080
- **Node Exporter**: http://localhost:9100

## Deployment

### Docker

```bash
# Build image
docker build -t ticket-classifier:latest .

# Run container
docker run ticket-classifier:latest
```

### Kubernetes

```bash
# Apply manifest
kubectl apply -f k8s/deployment.yaml

# Check deployment
kubectl get pods
kubectl logs -l app=pipeline
```

## CI/CD

The project uses GitHub Actions for continuous integration:

- **Lint Job**: Runs flake8 for code quality
- **Build Job**: Builds Docker image

Workflow triggers on push to `main` and `develop` branches.

## Development

### Code Formatting

```bash
# Format code with Black
black src/

# Lint with Ruff
ruff check src/
```

### Run Tests

```bash
pytest
```

## Models

Trained models are stored in the `models/` directory:
- `model.joblib` - Trained classifier
- `encoder.joblib` - Label encoder

## License

MIT License
