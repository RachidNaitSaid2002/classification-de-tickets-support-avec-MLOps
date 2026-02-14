FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY monitoring/ ./monitoring/
COPY models/ ./models/
COPY data/ ./data/
COPY chroma/ ./chroma/

# Create output directory
RUN mkdir -p /app/output

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.model_training"]
