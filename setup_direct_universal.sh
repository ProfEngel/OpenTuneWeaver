#!/bin/bash

# ============================================
# OpenTuneWeaver Universal Setup
# Version: 4.0 - OpenAI API Compatible
# Supports: OpenAI, OpenRouter, Ollama, LMStudio, etc.
# ============================================

set -e # Exit on error

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging Funktionen
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# ============================================
# Environment Variables for API Configuration
# ============================================

# Default values - can be overridden by environment variables
: ${OPENAI_API_BASE:="http://localhost:11434/v1"}
: ${OPENAI_API_KEY:="dummy-key"}
: ${OPENAI_MODEL_NAME:="llama3.2:latest"}
: ${OPENAI_API_TYPE:="ollama"}  # ollama, openai, openrouter, lmstudio, custom
: ${INSTALL_DIR:="/workspace/OpenTuneWeaver"}
: ${SKIP_SYSTEM_DEPS:="false"}
: ${SKIP_CUDA_CHECK:="false"}
: ${BUILD_LLAMACPP:="true"}
: ${SERVER_PORT:="8080"}
: ${SERVER_HOST:="0.0.0.0"}

# ============================================
# SCHRITT 1: System Info
# ============================================

log "${BLUE}========================================${NC}"
log "${BLUE}🚀 OpenTuneWeaver Universal Setup v4.0${NC}"
log "${BLUE}========================================${NC}"

# System Info
log "📊 System Information:"
echo "  Hostname: $(hostname)"
echo "  CPU: $(nproc) cores"
echo "  RAM: $(free -h | grep Mem | awk '{print $2}')"
if [ "$SKIP_CUDA_CHECK" != "true" ]; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
fi
echo "  Python: $(python3 --version 2>/dev/null || echo 'Python3 not found')"
echo "  Current Dir: $(pwd)"

# Display API Configuration
log "🔌 API Configuration:"
echo "  API Type: $OPENAI_API_TYPE"
echo "  API Base URL: $OPENAI_API_BASE"
echo "  Model: $OPENAI_MODEL_NAME"
echo "  Install Directory: $INSTALL_DIR"

# ============================================
# SCHRITT 2: System Dependencies
# ============================================

if [ "$SKIP_SYSTEM_DEPS" != "true" ]; then
    log "${BLUE}📦 Installing System Dependencies...${NC}"
    
    # Detect if we're root (typical in Docker)
    if [ "$EUID" -eq 0 ]; then
        APT_CMD="apt-get"
    else
        APT_CMD="sudo apt-get"
    fi
    
    # Update package list
    $APT_CMD update
    
    # Install dependencies in non-interactive mode
    DEBIAN_FRONTEND=noninteractive $APT_CMD install -y \
        build-essential \
        git \
        wget \
        curl \
        cmake \
        ninja-build \
        pkg-config \
        libssl-dev \
        libcurl4-openssl-dev \
        python3-pip \
        python3-dev \
        python3-venv \
        libreoffice \
        wkhtmltopdf \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-deu \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        fonts-liberation \
        htop \
        tmux \
        nano \
        unzip \
        2>/dev/null || warning "Some packages failed to install"
    
    log "✅ System dependencies installed"
else
    log "⏭️  Skipping system dependencies (SKIP_SYSTEM_DEPS=true)"
fi

# ============================================
# SCHRITT 3: CUDA Verification
# ============================================

if [ "$SKIP_CUDA_CHECK" != "true" ]; then
    log "${BLUE}🎮 Checking CUDA...${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        log "✅ CUDA is available"
        export CUDA_AVAILABLE=true
    else
        warning "No CUDA detected - CPU only mode"
        export CUDA_AVAILABLE=false
    fi
else
    log "⏭️  Skipping CUDA check (SKIP_CUDA_CHECK=true)"
    export CUDA_AVAILABLE=false
fi

# ============================================
# SCHRITT 4: Create Installation Directory
# ============================================

log "${BLUE}📁 Setting up installation directory...${NC}"

# Create parent directory if needed
mkdir -p "$(dirname "$INSTALL_DIR")"

# ============================================
# SCHRITT 5: Clone Repository
# ============================================

log "${BLUE}📥 Setting up OpenTuneWeaver Repository...${NC}"

if [ -d "$INSTALL_DIR" ]; then
    warning "OpenTuneWeaver directory exists, cleaning..."
    rm -rf "$INSTALL_DIR"
fi

# Clone repository
git clone https://github.com/ProfEngel/OpenTuneWeaver.git "$INSTALL_DIR"
cd "$INSTALL_DIR"

log "✅ Repository cloned to: $INSTALL_DIR"

# ============================================
# SCHRITT 6: Python Environment Setup
# ============================================

log "${BLUE}🐍 Setting up Python environment...${NC}"

# Upgrade pip
python3 -m pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f "requirements.txt" ]; then
    log "Installing packages from requirements.txt..."
    pip3 install -r requirements.txt
    log "✅ Repository requirements installed"
else
    error "requirements.txt not found!"
fi

# Install additional ML packages
log "Installing additional ML packages..."
pip3 install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || warning "Unsloth installation failed"

# Install xformers for CUDA if available
if [ "$CUDA_AVAILABLE" = true ]; then
    pip3 install --no-deps "xformers<0.0.27" || warning "xformers installation failed"
    log "✅ CUDA-specific packages installed"
fi

log "✅ Python environment setup complete"

# ============================================
# SCHRITT 7: Apply Path Fixes
# ============================================

log "${BLUE}🔧 Applying path fixes...${NC}"

# Backup original files
if [ -f "ui/app.py" ]; then
    cp ui/app.py ui/app.py.backup
fi

# Create symbolic link for pipeline
cd ui
if [ ! -L "pipeline" ]; then
    ln -sf ../pipeline pipeline
    log "✅ Created symbolic link ui/pipeline -> ../pipeline"
fi
cd ..

# Path corrections
if [ -f "ui/app.py" ]; then
    sed -i 's|"../pipeline/|"pipeline/|g' ui/app.py
    sed -i 's|Path("../pipeline")|Path("pipeline")|g' ui/app.py
    log "✅ Applied path corrections to ui/app.py"
fi

if [ -f "pipeline/run_pipeline.py" ]; then
    sed -i 's|Path("../|Path("|g' pipeline/run_pipeline.py
    log "✅ Fixed pipeline/run_pipeline.py paths"
fi

if [ -f "pipeline/config_loader.py" ]; then
    sed -i 's|Path.cwd().parent.parent.parent|Path.cwd().parent|g' pipeline/config_loader.py
    log "✅ Fixed pipeline/config_loader.py paths"
fi

# ============================================
# SCHRITT 8: Build llama.cpp (Optional)
# ============================================

if [ "$BUILD_LLAMACPP" = "true" ]; then
    log "${BLUE}🔨 Building llama.cpp...${NC}"
    
    cd "$INSTALL_DIR/pipeline/modules/06_finetuning"
    
    # Clone llama.cpp if not exists
    if [ ! -d "llama.cpp" ]; then
        log "Cloning llama.cpp..."
        git clone --recursive https://github.com/ggerganov/llama.cpp
    fi
    
    cd llama.cpp
    rm -rf build
    mkdir build
    cd build
    
    # Build based on CUDA availability
    if [ "$CUDA_AVAILABLE" = true ]; then
        log "Building llama.cpp with CUDA support..."
        cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    else
        log "Building llama.cpp CPU-only version..."
        cmake .. -DCMAKE_BUILD_TYPE=Release
    fi
    
    make -j$(nproc)
    
    if [ -f "bin/llama-cli" ] || [ -f "bin/main" ]; then
        log "✅ llama.cpp built successfully"
    else
        warning "llama.cpp build failed - continuing without GGUF support"
    fi
    
    cd "$INSTALL_DIR"
else
    log "⏭️  Skipping llama.cpp build (BUILD_LLAMACPP=false)"
fi

# ============================================
# SCHRITT 9: Create Universal Pipeline Configuration
# ============================================

log "${BLUE}📝 Creating universal pipeline configuration...${NC}"

# Determine base model based on API type
case "$OPENAI_API_TYPE" in
    "openai")
        DEFAULT_BASE_MODEL="gpt-4"
        ;;
    "openrouter")
        DEFAULT_BASE_MODEL="meta-llama/llama-3.2-3b-instruct"
        ;;
    *)
        DEFAULT_BASE_MODEL="unsloth/gemma-2b-it"
        ;;
esac

cat > "$INSTALL_DIR/pipeline/pipeline_config.json" << EOF
{
  "version": "4.0-universal",
  "created": "$(date -Iseconds)",
  "api_type": "${OPENAI_API_TYPE}",
  "tokens": {
    "hf_token": "${HF_TOKEN:-}",
    "hf_write_token": "${HF_WRITE_TOKEN:-}"
  },
  "api_configs": {
    "01_convert": {
      "use_openai_api": true,
      "openai_base_url": "${OPENAI_API_BASE}",
      "openai_api_key": "${OPENAI_API_KEY}",
      "openai_model_name": "${OPENAI_MODEL_NAME}",
      "temperature": 0.1,
      "max_tokens": 4096,
      "timeout": 300
    },
    "02_genwiki": {
      "use_openai_api": true,
      "openai_base_url": "${OPENAI_API_BASE}",
      "openai_api_key": "${OPENAI_API_KEY}",
      "openai_model_name": "${OPENAI_MODEL_NAME}",
      "temperature": 0.3,
      "max_tokens": 4096,
      "timeout": 300
    },
    "03_instructQA": {
      "use_openai_api": true,
      "openai_base_url": "${OPENAI_API_BASE}",
      "openai_api_key": "${OPENAI_API_KEY}",
      "openai_model_name": "${OPENAI_MODEL_NAME}",
      "temperature": 0.7,
      "max_tokens": 4096,
      "timeout": 300
    },
    "05_bmcreator": {
      "use_openai_api": true,
      "openai_base_url": "${OPENAI_API_BASE}",
      "openai_api_key": "${OPENAI_API_KEY}",
      "openai_model_name": "${OPENAI_MODEL_NAME}",
      "temperature": 0.5,
      "max_tokens": 4096,
      "timeout": 300
    }
  },
  "finetuning": {
    "model_name": "${FINETUNING_MODEL_NAME:-OTW-Model}",
    "base_model": "${FINETUNING_BASE_MODEL:-$DEFAULT_BASE_MODEL}",
    "hf_repo_id": "${HF_REPO_ID:-user/OTW-Model}",
    "dataset_path": "INPUT/dataset.json",
    "chat_template": "${CHAT_TEMPLATE:-gemma}",
    "custom_model_dir": "CustomModel",
    "max_seq_length": ${MAX_SEQ_LENGTH:-2048},
    "load_in_4bit": ${LOAD_IN_4BIT:-true},
    "full_finetuning": ${FULL_FINETUNING:-false},
    "lora_r": ${LORA_R:-8},
    "lora_alpha": ${LORA_ALPHA:-8},
    "lora_dropout": ${LORA_DROPOUT:-0},
    "per_device_train_batch_size": ${BATCH_SIZE:-1},
    "gradient_accumulation_steps": ${GRAD_ACCUM:-8},
    "warmup_steps": ${WARMUP_STEPS:-100},
    "num_train_epochs": ${NUM_EPOCHS:-3},
    "learning_rate": ${LEARNING_RATE:-5e-5},
    "save_lora": ${SAVE_LORA:-true},
    "save_merged": ${SAVE_MERGED:-true},
    "save_gguf": ${SAVE_GGUF:-false}
  },
  "benchmark": {
    "mode": "${BENCHMARK_MODE:-comparison}",
    "evaluator": {
      "type": "api",
      "api_base_url": "${OPENAI_API_BASE}",
      "api_key": "${OPENAI_API_KEY}",
      "model": "${OPENAI_MODEL_NAME}"
    }
  },
  "pipeline": {
    "auto_cleanup": ${AUTO_CLEANUP:-false},
    "verbose": ${VERBOSE:-true},
    "continue_on_error": ${CONTINUE_ON_ERROR:-true}
  }
}
EOF

log "✅ Universal configuration created"

# ============================================
# SCHRITT 10: Create Directory Structure
# ============================================

log "${BLUE}📁 Creating directory structure...${NC}"

cd "$INSTALL_DIR"

# Create all necessary directories
mkdir -p pipeline/data/OUTPUT
mkdir -p pipeline/modules/01_convert/{UPLOAD,INPUT,OUTPUT}
mkdir -p pipeline/modules/02_wiki/{INPUT,OUTPUT}
mkdir -p pipeline/modules/03_instructQA/{INPUT,OUTPUT}
mkdir -p pipeline/modules/04_format/{INPUT,OUTPUT}
mkdir -p pipeline/modules/05_bmcreator/{INPUT,BENCHMARKFRAGEN}
mkdir -p pipeline/modules/06_finetuning/{INPUT,CustomModel,results}
mkdir -p pipeline/modules/07_benchmark/{BENCHMARKFRAGEN,OUTPUT}
mkdir -p viewer/images

log "✅ Directory structure created"

# ============================================
# SCHRITT 11: Create Startup Scripts
# ============================================

log "${BLUE}📝 Creating startup scripts...${NC}"

# Main startup script
cat > "$INSTALL_DIR/start_otw.sh" << EOF
#!/bin/bash

export OPENAI_API_BASE="${OPENAI_API_BASE}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export OPENAI_MODEL_NAME="${OPENAI_MODEL_NAME}"
export OPENAI_API_TYPE="${OPENAI_API_TYPE}"

echo "🚀 Starting OpenTuneWeaver..."
echo "📡 API Type: \${OPENAI_API_TYPE}"
echo "🔗 API Base: \${OPENAI_API_BASE}"
echo "🤖 Model: \${OPENAI_MODEL_NAME}"

cd "$INSTALL_DIR"

# Verify UI structure
cd ui
if [ ! -L "pipeline" ]; then
    ln -sf ../pipeline pipeline
fi

# Start UI
echo "Starting OpenTuneWeaver UI on http://${SERVER_HOST}:${SERVER_PORT}"
python3 app.py --server_name ${SERVER_HOST} --server_port ${SERVER_PORT}
EOF

chmod +x "$INSTALL_DIR/start_otw.sh"

# Docker entrypoint script
cat > "$INSTALL_DIR/docker-entrypoint.sh" << 'EOF'
#!/bin/bash

# Allow dynamic configuration through environment variables
if [ -n "$OPENAI_API_BASE" ]; then
    echo "🔄 Updating API configuration from environment..."
    
    # Update config file with environment variables
    python3 - <<PYTHON
import json
import os

config_path = '/workspace/OpenTuneWeaver/pipeline/pipeline_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

# Update API configs
for module in ['01_convert', '02_genwiki', '03_instructQA', '05_bmcreator']:
    if module in config['api_configs']:
        config['api_configs'][module]['openai_base_url'] = os.environ.get('OPENAI_API_BASE', config['api_configs'][module]['openai_base_url'])
        config['api_configs'][module]['openai_api_key'] = os.environ.get('OPENAI_API_KEY', config['api_configs'][module]['openai_api_key'])
        config['api_configs'][module]['openai_model_name'] = os.environ.get('OPENAI_MODEL_NAME', config['api_configs'][module]['openai_model_name'])

# Update benchmark config
config['benchmark']['evaluator']['api_base_url'] = os.environ.get('OPENAI_API_BASE', config['benchmark']['evaluator']['api_base_url'])
config['benchmark']['evaluator']['api_key'] = os.environ.get('OPENAI_API_KEY', config['benchmark']['evaluator']['api_key'])
config['benchmark']['evaluator']['model'] = os.environ.get('OPENAI_MODEL_NAME', config['benchmark']['evaluator']['model'])

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Configuration updated")
PYTHON
fi

# Execute the main startup
exec /workspace/OpenTuneWeaver/start_otw.sh
EOF

chmod +x "$INSTALL_DIR/docker-entrypoint.sh"

# Health check script
cat > "$INSTALL_DIR/healthcheck.sh" << EOF
#!/bin/bash

# Check if the UI is responding
curl -f http://localhost:${SERVER_PORT}/ > /dev/null 2>&1
if [ \$? -eq 0 ]; then
    echo "✅ OpenTuneWeaver is healthy"
    exit 0
else
    echo "❌ OpenTuneWeaver is not responding"
    exit 1
fi
EOF

chmod +x "$INSTALL_DIR/healthcheck.sh"

log "✅ Startup scripts created"

# ============================================
# SCHRITT 12: Create Docker Support Files
# ============================================

log "${BLUE}🐳 Creating Docker support files...${NC}"

# Create Dockerfile
cat > "$INSTALL_DIR/Dockerfile" << EOF
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV INSTALL_DIR=/workspace/OpenTuneWeaver
ENV SKIP_SYSTEM_DEPS=false
ENV BUILD_LLAMACPP=true

# Default API configuration (can be overridden)
ENV OPENAI_API_TYPE=ollama
ENV OPENAI_API_BASE=http://localhost:11434/v1
ENV OPENAI_API_KEY=dummy-key
ENV OPENAI_MODEL_NAME=llama3.2:latest
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3 python3-pip git curl wget \\
    && rm -rf /var/lib/apt/lists/*

# Copy setup script
COPY setup_direct_universal.sh /setup.sh
RUN chmod +x /setup.sh

# Run setup
RUN /setup.sh

# Expose port
EXPOSE 8080

# Set working directory
WORKDIR /workspace/OpenTuneWeaver

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD /workspace/OpenTuneWeaver/healthcheck.sh

# Entry point
ENTRYPOINT ["/workspace/OpenTuneWeaver/docker-entrypoint.sh"]
EOF

# Create docker-compose.yml
cat > "$INSTALL_DIR/docker-compose.yml" << EOF
version: '3.8'

services:
  opentuneweaver:
    build: .
    image: opentuneweaver:latest
    container_name: opentuneweaver
    ports:
      - "8080:8080"
    environment:
      # API Configuration - adjust these for your provider
      - OPENAI_API_TYPE=ollama  # ollama, openai, openrouter, lmstudio, custom
      - OPENAI_API_BASE=http://host.docker.internal:11434/v1
      - OPENAI_API_KEY=\${OPENAI_API_KEY:-dummy-key}
      - OPENAI_MODEL_NAME=\${OPENAI_MODEL_NAME:-llama3.2:latest}
      
      # Optional: HuggingFace tokens
      - HF_TOKEN=\${HF_TOKEN:-}
      - HF_WRITE_TOKEN=\${HF_WRITE_TOKEN:-}
      
      # Server configuration
      - SERVER_HOST=0.0.0.0
      - SERVER_PORT=8080
      
      # Pipeline configuration
      - VERBOSE=true
      - CONTINUE_ON_ERROR=true
    volumes:
      - ./data:/workspace/OpenTuneWeaver/pipeline/data
      - ./models:/workspace/OpenTuneWeaver/models
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    networks:
      - otw-network

  # Optional: Ollama service (comment out if using external API)
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped
    networks:
      - otw-network

networks:
  otw-network:
    driver: bridge

volumes:
  ollama_data:
EOF

# Create .env.example
cat > "$INSTALL_DIR/.env.example" << EOF
# OpenTuneWeaver Environment Configuration

# API Provider Configuration
# Options: ollama, openai, openrouter, lmstudio, custom
OPENAI_API_TYPE=ollama

# API Endpoints (examples for different providers)
# Ollama (local): http://localhost:11434/v1
# OpenAI: https://api.openai.com/v1
# OpenRouter: https://openrouter.ai/api/v1
# LM Studio: http://localhost:1234/v1
OPENAI_API_BASE=http://localhost:11434/v1

# API Key (use actual key for OpenAI/OpenRouter)
OPENAI_API_KEY=dummy-key

# Model Name (examples)
# Ollama: llama3.2:latest, gemma2:27b, mistral:latest
# OpenAI: gpt-4, gpt-3.5-turbo
# OpenRouter: meta-llama/llama-3.2-3b-instruct
OPENAI_MODEL_NAME=llama3.2:latest

# Optional: HuggingFace Tokens
HF_TOKEN=
HF_WRITE_TOKEN=

# Finetuning Configuration
FINETUNING_MODEL_NAME=OTW-Model
FINETUNING_BASE_MODEL=unsloth/gemma-2b-it
MAX_SEQ_LENGTH=2048
NUM_EPOCHS=3
LEARNING_RATE=5e-5
EOF

log "✅ Docker support files created"

# ============================================
# SCHRITT 13: Test Installation
# ============================================

log "${BLUE}🧪 Testing installation...${NC}"

cd "$INSTALL_DIR"

# Test Python imports
python3 -c "
import sys
try:
    import torch
    import transformers
    import gradio
    print('✅ Core packages imported successfully')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  Transformers: {transformers.__version__}')
    print(f'  Gradio: {gradio.__version__}')
    if torch.cuda.is_available():
        print(f'  CUDA: Available - {torch.cuda.get_device_name(0)}')
    else:
        print(f'  CUDA: Not available (CPU mode)')
except ImportError as e:
    print(f'⚠️  Import warning: {e}')
"

# Test app.py syntax
cd ui && python3 -m py_compile app.py 2>/dev/null && log "✅ app.py syntax OK" || warning "app.py syntax check failed"
cd ..

# ============================================
# SCHRITT 14: Installation Summary
# ============================================

log "${GREEN}========================================${NC}"
log "${GREEN}✅ Installation Complete!${NC}"
log "${GREEN}========================================${NC}"

echo ""
echo "📁 Installation directory: $INSTALL_DIR"
echo "🔌 API Configuration:"
echo "   Type: $OPENAI_API_TYPE"
echo "   Base URL: $OPENAI_API_BASE"
echo "   Model: $OPENAI_MODEL_NAME"
echo ""
echo "🚀 Quick Start:"
echo "   Direct:  $INSTALL_DIR/start_otw.sh"
echo "   Docker:  cd $INSTALL_DIR && docker-compose up"
echo ""
echo "🌐 Access URL: http://${SERVER_HOST}:${SERVER_PORT}"
echo ""
echo "📝 Configuration files:"
echo "   Pipeline: $INSTALL_DIR/pipeline/pipeline_config.json"
echo "   Docker:   $INSTALL_DIR/.env (copy from .env.example)"
echo ""
echo "🔧 To use different API providers, set environment variables:"
echo "   export OPENAI_API_TYPE=openai"
echo "   export OPENAI_API_BASE=https://api.openai.com/v1"
echo "   export OPENAI_API_KEY=your-api-key"
echo "   export OPENAI_MODEL_NAME=gpt-4"
echo ""

# Auto-start check (only if not in Docker/CI environment)
if [ -t 0 ] && [ -z "$CI" ] && [ -z "$DOCKER_CONTAINER" ]; then
    read -t 10 -p "🚀 Start OpenTuneWeaver now? (y/n) " -n 1 -r START_NOW || true
    echo
    if [[ $START_NOW =~ ^[Yy]$ ]]; then
        log "Starting OpenTuneWeaver..."
        cd "$INSTALL_DIR"
        ./start_otw.sh
    fi
fi