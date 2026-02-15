# Use slim Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed, can add more later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (excluding what's in .dockerignore)
COPY . .

# Ensure output directory exists
RUN mkdir -p /app/output

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "src.model_training"]
