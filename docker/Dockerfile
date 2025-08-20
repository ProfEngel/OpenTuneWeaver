# OpenTuneWeaver Universal Docker Image (Standard)
# Multi-architecture support: linux/amd64, linux/arm64
FROM python:3.11-slim-bookworm

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Labels
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="Prof. Dr. Mathias Engel" \
      org.opencontainers.image.url="https://github.com/ProfEngel/OpenTuneWeaver" \
      org.opencontainers.image.documentation="https://github.com/ProfEngel/OpenTuneWeaver/blob/main/README.md" \
      org.opencontainers.image.source="https://github.com/ProfEngel/OpenTuneWeaver" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="OpenTuneWeaver" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="OpenTuneWeaver" \
      org.opencontainers.image.description="Universal LLM Fine-tuning Pipeline with External API Support"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential tools
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    # Development tools
    cmake \
    ninja-build \
    pkg-config \
    # Libraries
    libssl-dev \
    libcurl4-openssl-dev \
    libgomp1 \
    # Document processing
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    wkhtmltopdf \
    # Python development
    python3-dev \
    # Utilities
    jq \
    htop \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install base packages
RUN pip install --upgrade pip setuptools wheel

# Clone OpenTuneWeaver repository
RUN git clone https://github.com/ProfEngel/OpenTuneWeaver.git /app/OpenTuneWeaver \
    && cd /app/OpenTuneWeaver \
    && git config --global --add safe.directory /app/OpenTuneWeaver

WORKDIR /app/OpenTuneWeaver

# Copy and install Python requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional packages (without CUDA)
RUN pip install --no-cache-dir \
    gradio>=5.0.0 \
    transformers>=4.40.0 \
    datasets>=2.14.0 \
    peft>=0.10.0 \
    accelerate>=0.25.0 \
    bitsandbytes>=0.42.0 \
    sentencepiece \
    protobuf \
    && pip install --no-cache-dir "unsloth @ git+https://github.com/unslothai/unsloth.git" || true

# Create directory structure
RUN mkdir -p \
    /app/OpenTuneWeaver/pipeline/data/OUTPUT \
    /app/OpenTuneWeaver/pipeline/modules/01_convert/{UPLOAD,INPUT,OUTPUT} \
    /app/OpenTuneWeaver/pipeline/modules/02_wiki/{INPUT,OUTPUT} \
    /app/OpenTuneWeaver/pipeline/modules/03_instructQA/{INPUT,OUTPUT} \
    /app/OpenTuneWeaver/pipeline/modules/04_format/{INPUT,OUTPUT} \
    /app/OpenTuneWeaver/pipeline/modules/05_bmcreator/{INPUT,BENCHMARKFRAGEN} \
    /app/OpenTuneWeaver/pipeline/modules/06_finetuning/{INPUT,CustomModel,results} \
    /app/OpenTuneWeaver/pipeline/modules/07_benchmark/{BENCHMARKFRAGEN,OUTPUT} \
    /app/OpenTuneWeaver/viewer/images \
    /app/OpenTuneWeaver/logs \
    /app/OpenTuneWeaver/cache

# Setup app.py
RUN if [ -f "ui/app_new.py" ]; then \
        cp ui/app_new.py app.py; \
    elif [ -f "ui/app.py" ]; then \
        cp ui/app.py app.py; \
    elif [ -f "gradio_app/app.py" ]; then \
        cp gradio_app/app.py app.py; \
    fi

# Create healthcheck script
RUN cat > /app/healthcheck.sh << 'EOF'
#!/bin/bash
curl -f http://localhost:${PORT:-8080}/ || exit 1
EOF
RUN chmod +x /app/healthcheck.sh

# Create entrypoint script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "================================================"
echo " OpenTuneWeaver Universal - Docker Container"
echo " Version: ${VERSION:-latest}"
echo "================================================"

# Setup from environment variables
setup_env() {
    export API_BASE_URL="${OPENAI_API_BASE:-${API_BASE_URL:-http://host.docker.internal:11434/v1}}"
    export API_KEY="${OPENAI_API_KEY:-${API_KEY:-ollama}}"
    export MODEL_NAME="${OPENAI_MODEL:-${MODEL_NAME:-llama3.2:3b}}"
    
    echo "Configuration:"
    echo "  API URL: $API_BASE_URL"
    echo "  Model: $MODEL_NAME"
    echo "  Port: ${PORT:-8080}"
    echo ""
}

# Create configuration file
create_config() {
    cat > /app/OpenTuneWeaver/pipeline/pipeline_config.json << EOL
{
  "version": "docker-universal",
  "api_configs": {
    "01_convert": {
      "use_openai_api": true,
      "openai_base_url": "${API_BASE_URL}",
      "openai_api_key": "${API_KEY}",
      "openai_model_name": "${MODEL_NAME}",
      "temperature": 0.1
    },
    "02_genwiki": {
      "use_openai_api": true,
      "openai_base_url": "${API_BASE_URL}",
      "openai_api_key": "${API_KEY}",
      "openai_model_name": "${MODEL_NAME}",
      "temperature": 0.3
    },
    "03_instructQA": {
      "use_openai_api": true,
      "openai_base_url": "${API_BASE_URL}",
      "openai_api_key": "${API_KEY}",
      "openai_model_name": "${MODEL_NAME}",
      "temperature": 0.7
    },
    "05_bmcreator": {
      "use_openai_api": true,
      "openai_base_url": "${API_BASE_URL}",
      "openai_api_key": "${API_KEY}",
      "openai_model_name": "${MODEL_NAME}",
      "temperature": 0.5
    }
  }
}
EOL
}

# Test API connection
test_api() {
    echo "Testing API connection..."
    if curl -s --max-time 5 "${API_BASE_URL%/v1}/api/tags" > /dev/null 2>&1 || \
       curl -s --max-time 5 -H "Authorization: Bearer $API_KEY" "${API_BASE_URL}/models" > /dev/null 2>&1; then
        echo "✅ API connection successful"
    else
        echo "⚠️  Could not verify API connection (service might not be ready yet)"
    fi
    echo ""
}

# Main execution
setup_env
create_config
test_api

# Start application
cd /app/OpenTuneWeaver
echo "Starting OpenTuneWeaver on port ${PORT:-8080}..."
echo "================================================"

exec python3 app.py \
    --server_name 0.0.0.0 \
    --server_port ${PORT:-8080} \
    --share ${GRADIO_SHARE:-false}
EOF

RUN chmod +x /app/entrypoint.sh

# Expose ports
EXPOSE 8080 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Volumes for persistent data
VOLUME ["/app/OpenTuneWeaver/pipeline/data", \
        "/app/OpenTuneWeaver/pipeline/modules", \
        "/app/OpenTuneWeaver/logs", \
        "/app/OpenTuneWeaver/cache"]

# Default environment variables
ENV PORT=8080 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SHARE=false

# Run as non-root user (optional, comment out if issues)
# RUN useradd -m -u 1000 otw && chown -R otw:otw /app
# USER otw

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]