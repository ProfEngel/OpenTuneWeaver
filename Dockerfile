# OpenTuneWeaver Universal Docker Image (Dataset Generator)
FROM python:3.11-slim

# Labels
LABEL org.opencontainers.image.title="OpenTuneWeaver" \
      org.opencontainers.image.description="Universal Dataset Generator Pipeline"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8080

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    # Document processing
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# 5. Copy requirements file
COPY requirements.txt /app/requirements.txt

# Force install PyTorch (CPU only) BEFORE other requirements so docling doesn't pull the 900+ MB CUDA version
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6. Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy codebase
COPY . /app/

# Expose Grado port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Entrypoint to run Gradio app
CMD ["python", "app.py"]
