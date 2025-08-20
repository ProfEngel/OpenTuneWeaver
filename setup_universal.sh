#!/bin/bash

# ============================================
# OpenTuneWeaver Universal Setup (External API)
# Version: 1.0-universal
# ============================================
# This script sets up OpenTuneWeaver to use existing
# OpenAI-compatible API endpoints (Ollama, vLLM, LocalAI, etc.)
# without installing Ollama locally
# ============================================

set -e # Exit on error

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
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

info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

# ============================================
# Configuration Detection & Setup
# ============================================

detect_api_config() {
    log "${BLUE}🔍 Detecting API Configuration...${NC}"
    
    # Check for environment variables
    if [ -n "$OPENAI_API_BASE" ] || [ -n "$OPENAI_BASE_URL" ]; then
        export API_BASE_URL="${OPENAI_API_BASE:-$OPENAI_BASE_URL}"
        info "Found API base URL in environment: $API_BASE_URL"
    else
        # Ask user for API configuration
        echo ""
        echo "Please provide your OpenAI-compatible API configuration:"
        read -p "API Base URL (e.g., http://localhost:11434/v1): " API_BASE_URL
        export API_BASE_URL
    fi
    
    # API Key (optional for local services)
    if [ -n "$OPENAI_API_KEY" ]; then
        export API_KEY="$OPENAI_API_KEY"
        info "Found API key in environment"
    else
        read -p "API Key (press Enter for 'ollama' or local services): " API_KEY
        export API_KEY="${API_KEY:-ollama}"
    fi
    
    # Model name
    if [ -n "$OPENAI_MODEL" ] || [ -n "$MODEL_NAME" ]; then
        export MODEL_NAME="${OPENAI_MODEL:-$MODEL_NAME}"
        info "Found model name in environment: $MODEL_NAME"
    else
        echo ""
        echo "Common model names:"
        echo "  - For Ollama: gemma3:12b-it-qat, llama3.2:3b, mistral, etc."
        echo "  - For vLLM: meta-llama/Llama-3-8B, etc."
        echo "  - For OpenAI: gpt-4, gpt-3.5-turbo, etc."
        read -p "Model name: " MODEL_NAME
        export MODEL_NAME
    fi
    
    # Test API connection
    log "Testing API connection..."
    if curl -s "${API_BASE_URL%/v1}/api/tags" > /dev/null 2>&1 || \
       curl -s -H "Authorization: Bearer $API_KEY" "${API_BASE_URL}/models" > /dev/null 2>&1; then
        log "✅ API connection successful!"
    else
        warning "Could not verify API connection. Please ensure your API service is running."
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# ============================================
# SCHRITT 1: System Info und Vorbereitung
# ============================================

log "${BLUE}========================================${NC}"
log "${BLUE}🚀 OpenTuneWeaver Universal Setup v1.0${NC}"
log "${BLUE}========================================${NC}"

# System Info
log "📊 System Information:"
echo "  Hostname: $(hostname)"
echo "  CPU: $(nproc) cores"
echo "  RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo "  Python: $(python3 --version)"
echo "  Current Dir: $(pwd)"
echo ""

# Detect if running in container
if [ -f /.dockerenv ]; then
    info "🐳 Running in Docker container"
    export IN_DOCKER=true
else
    export IN_DOCKER=false
fi

# Detect if running in RunPod
if [ -n "$RUNPOD_POD_ID" ]; then
    info "☁️ Running in RunPod environment"
    export IN_RUNPOD=true
    export WORKSPACE_DIR="/workspace"
else
    export IN_RUNPOD=false
    export WORKSPACE_DIR="$(pwd)"
fi

# ============================================
# SCHRITT 2: API Configuration
# ============================================

detect_api_config

# ============================================
# SCHRITT 3: System Dependencies
# ============================================

log "${BLUE}📦 Installing System Dependencies...${NC}"

# Check if we have sudo permissions
if [ "$IN_DOCKER" = true ]; then
    apt-get update
    INSTALL_CMD="apt-get install -y"
else
    if command -v sudo &> /dev/null; then
        sudo apt-get update
        INSTALL_CMD="sudo apt-get install -y"
    else
        warning "No sudo available, trying without..."
        apt-get update
        INSTALL_CMD="apt-get install -y"
    fi
fi

$INSTALL_CMD \
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
    jq || warning "Some packages might have failed to install"

log "✅ System dependencies installed"

# ============================================
# SCHRITT 4: NVIDIA/CUDA Verification
# ============================================

log "${BLUE}🎮 Checking CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    log "✅ CUDA is available"
    export CUDA_AVAILABLE=true
else
    warning "No CUDA detected - CPU only mode"
    export CUDA_AVAILABLE=false
fi

# ============================================
# SCHRITT 5: Clone Repository
# ============================================

log "${BLUE}📥 Setting up OpenTuneWeaver Repository...${NC}"

cd "$WORKSPACE_DIR"

# Check if we're already in OpenTuneWeaver directory
if [ -f "requirements.txt" ] && [ -d "pipeline" ]; then
    info "Already in OpenTuneWeaver directory"
    export OTW_DIR="$WORKSPACE_DIR"
else
    # Clone if not exists
    if [ ! -d "OpenTuneWeaver" ]; then
        git clone https://github.com/ProfEngel/OpenTuneWeaver.git
        cd OpenTuneWeaver
        export OTW_DIR="$WORKSPACE_DIR/OpenTuneWeaver"
    else
        cd OpenTuneWeaver
        git pull || warning "Could not update repository"
        export OTW_DIR="$WORKSPACE_DIR/OpenTuneWeaver"
    fi
fi

log "✅ Repository ready at: $OTW_DIR"

# ============================================
# SCHRITT 6: Python Environment Setup
# ============================================

log "${BLUE}🐍 Setting up Python environment...${NC}"

# Upgrade pip first
python3 -m pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f "requirements.txt" ]; then
    log "Installing packages from requirements.txt..."
    pip3 install -r requirements.txt
    log "✅ Repository requirements installed"
else
    error "requirements.txt not found!"
fi

# Install additional packages
log "Installing additional ML packages..."
pip3 install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || warning "Unsloth installation failed"

# Install xformers for CUDA
if [ "$CUDA_AVAILABLE" = true ]; then
    pip3 install --no-deps "xformers<0.0.27" || warning "xformers installation failed"
    log "✅ CUDA-specific packages installed"
fi

log "✅ Python environment setup complete"

# ============================================
# SCHRITT 7: Setup Directory Structure
# ============================================

log "${BLUE}📁 Creating directory structure...${NC}"

cd "$OTW_DIR"

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
mkdir -p logs
mkdir -p cache

# Setup app.py if needed
if [ ! -f "app.py" ]; then
    if [ -f "ui/app_new.py" ]; then
        cp ui/app_new.py app.py
        log "✅ app.py created from app_new.py"
    elif [ -f "ui/app.py" ]; then
        cp ui/app.py app.py
        log "✅ app.py created from ui/app.py"
    fi
fi

log "✅ Directory structure created"

# ============================================
# SCHRITT 8: Build llama.cpp (Optional)
# ============================================

log "${BLUE}🔨 Building llama.cpp (optional)...${NC}"

cd "$OTW_DIR/pipeline/modules/06_finetuning"

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
    log "Building llama.cpp CPU-only..."
    cmake .. -DCMAKE_BUILD_TYPE=Release
fi

make -j$(nproc) || warning "llama.cpp build failed"

cd "$OTW_DIR"

# ============================================
# SCHRITT 9: Create Configuration Files
# ============================================

log "${BLUE}📝 Creating configuration files...${NC}"

# Create main pipeline configuration
cat > "$OTW_DIR/pipeline/pipeline_config.json" << EOF
{
  "version": "1.0-universal",
  "created": "$(date -Iseconds)",
  "api_provider": "external",
  "tokens": {
    "hf_token": "${HF_TOKEN:-}",
    "hf_write_token": "${HF_WRITE_TOKEN:-}"
  },
  "api_configs": {
    "_comment": "Universal API configuration for external OpenAI-compatible services",
    "01_convert": {
      "use_openai_api": true,
      "openai_base_url": "${API_BASE_URL}",
      "openai_api_key": "${API_KEY}",
      "openai_model_name": "${MODEL_NAME}",
      "temperature": 0.1,
      "max_tokens": 4096
    },
    "02_genwiki": {
      "use_openai_api": true,
      "openai_base_url": "${API_BASE_URL}",
      "openai_api_key": "${API_KEY}",
      "openai_model_name": "${MODEL_NAME}",
      "temperature": 0.3,
      "max_tokens": 4096
    },
    "03_instructQA": {
      "use_openai_api": true,
      "openai_base_url": "${API_BASE_URL}",
      "openai_api_key": "${API_KEY}",
      "openai_model_name": "${MODEL_NAME}",
      "temperature": 0.7,
      "max_tokens": 2048
    },
    "05_bmcreator": {
      "use_openai_api": true,
      "openai_base_url": "${API_BASE_URL}",
      "openai_api_key": "${API_KEY}",
      "openai_model_name": "${MODEL_NAME}",
      "temperature": 0.5,
      "max_tokens": 2048
    }
  },
  "finetuning": {
    "model_name": "OTW-Model",
    "base_model": "unsloth/gemma-3n-E2B-it",
    "hf_repo_id": "user/OTW-Model",
    "dataset_path": "INPUT/dataset.json",
    "chat_template": "gemma",
    "custom_model_dir": "CustomModel",
    "max_seq_length": 2048,
    "load_in_4bit": true,
    "full_finetuning": false,
    "lora_r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0,
    "bias": "none",
    "random_state": 3407,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 100,
    "num_train_epochs": 3,
    "max_steps": -1,
    "learning_rate": 5e-5,
    "logging_steps": 5,
    "optim": "adamw_8bit",
    "weight_decay": 0.03,
    "lr_scheduler_type": "cosine",
    "seed": 3407,
    "save_lora": true,
    "save_merged": true,
    "save_gguf": true,
    "upload_to_hf": false,
    "gguf_quantizations": ["q8_0", "q4_k_m"],
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_new_tokens": 128
  },
  "benchmark": {
    "mode": "comparison",
    "pre_model": {
      "name": "unsloth/gemma-3n-E2B-it",
      "type": "transformers",
      "load_in_4bit": false,
      "max_seq_length": 2048
    },
    "post_model": {
      "name": "CustomModel/OTW-Model",
      "type": "local",
      "load_in_4bit": false,
      "max_seq_length": 2048
    },
    "evaluator": {
      "type": "api",
      "api_base_url": "${API_BASE_URL}",
      "api_key": "${API_KEY}",
      "model": "${MODEL_NAME}"
    },
    "questions_file": "BENCHMARKFRAGEN/benchmark_fragen_complete.json",
    "max_new_tokens": 256,
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
  }
}
EOF

# Create .env file for environment variables
cat > "$OTW_DIR/.env" << EOF
# OpenTuneWeaver Universal Configuration
# Generated: $(date)

# API Configuration
OPENAI_API_BASE=${API_BASE_URL}
OPENAI_API_KEY=${API_KEY}
OPENAI_MODEL=${MODEL_NAME}

# Alternative names (for compatibility)
API_BASE_URL=${API_BASE_URL}
API_KEY=${API_KEY}
MODEL_NAME=${MODEL_NAME}

# HuggingFace Tokens (optional)
HF_TOKEN=${HF_TOKEN:-}
HF_WRITE_TOKEN=${HF_WRITE_TOKEN:-}

# System Configuration
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=${GRADIO_PORT:-8080}

# Paths
OTW_DIR=${OTW_DIR}
WORKSPACE_DIR=${WORKSPACE_DIR}

# Feature Flags
USE_EXTERNAL_API=true
USE_OLLAMA=false
ENABLE_TELEMETRY=${ENABLE_TELEMETRY:-true}
ENABLE_CACHE=${ENABLE_CACHE:-true}
EOF

log "✅ Configuration files created"

# ============================================
# SCHRITT 10: Create Startup Scripts
# ============================================

log "${BLUE}📝 Creating startup scripts...${NC}"

# Main startup script
cat > "$WORKSPACE_DIR/start_otw.sh" << 'EOF'
#!/bin/bash

echo "🚀 Starting OpenTuneWeaver Universal..."

# Load environment variables
if [ -f "$OTW_DIR/.env" ]; then
    export $(grep -v '^#' "$OTW_DIR/.env" | xargs)
    echo "✅ Environment variables loaded"
fi

# Verify API configuration
if [ -z "$API_BASE_URL" ]; then
    echo "❌ Error: API_BASE_URL not configured!"
    echo "Please set your API configuration in .env file or environment variables"
    exit 1
fi

echo "📡 Using API: $API_BASE_URL"
echo "🤖 Model: $MODEL_NAME"

# Test API connection
echo "Testing API connection..."
if curl -s "${API_BASE_URL%/v1}/api/tags" > /dev/null 2>&1 || \
   curl -s -H "Authorization: Bearer $API_KEY" "${API_BASE_URL}/models" > /dev/null 2>&1; then
    echo "✅ API connection successful"
else
    echo "⚠️ Warning: Could not verify API connection"
    echo "Make sure your API service is running at: $API_BASE_URL"
fi

# Start OpenTuneWeaver
cd "$OTW_DIR"
echo "Starting OpenTuneWeaver UI on port ${GRADIO_PORT:-8080}..."
python3 app.py --server_name 0.0.0.0 --server_port ${GRADIO_PORT:-8080}
EOF

chmod +x "$WORKSPACE_DIR/start_otw.sh"

# Configuration helper script
cat > "$WORKSPACE_DIR/configure_api.sh" << 'EOF'
#!/bin/bash

echo "🔧 OpenTuneWeaver API Configuration Helper"
echo "========================================="
echo ""
echo "This script helps you configure the API connection"
echo ""

# Function to test API
test_api() {
    local url=$1
    local key=$2
    
    if curl -s "${url%/v1}/api/tags" > /dev/null 2>&1; then
        echo "✅ Ollama-style API detected"
        return 0
    elif curl -s -H "Authorization: Bearer $key" "${url}/models" > /dev/null 2>&1; then
        echo "✅ OpenAI-style API detected"
        return 0
    else
        echo "❌ Could not connect to API"
        return 1
    fi
}

# Get API URL
echo "Enter your API base URL:"
echo "Examples:"
echo "  - Ollama: http://localhost:11434/v1"
echo "  - vLLM: http://localhost:8000/v1"
echo "  - OpenAI: https://api.openai.com/v1"
echo ""
read -p "API URL: " API_URL

# Get API Key
echo ""
echo "Enter your API key (press Enter for 'ollama' or if not required):"
read -s -p "API Key: " API_KEY
echo ""
API_KEY=${API_KEY:-ollama}

# Test connection
echo ""
echo "Testing connection..."
if test_api "$API_URL" "$API_KEY"; then
    echo ""
    
    # Get model name
    echo "Enter the model name to use:"
    read -p "Model: " MODEL
    
    # Update .env file
    cat > "$OTW_DIR/.env" << EOL
# OpenTuneWeaver API Configuration
# Updated: $(date)

OPENAI_API_BASE=$API_URL
OPENAI_API_KEY=$API_KEY
OPENAI_MODEL=$MODEL

API_BASE_URL=$API_URL
API_KEY=$API_KEY
MODEL_NAME=$MODEL

# Other settings
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=8080
OTW_DIR=$OTW_DIR
WORKSPACE_DIR=$WORKSPACE_DIR
USE_EXTERNAL_API=true
EOL
    
    echo ""
    echo "✅ Configuration saved to $OTW_DIR/.env"
    echo ""
    echo "You can now start OpenTuneWeaver with: ./start_otw.sh"
else
    echo ""
    echo "Failed to connect to API. Please check your settings."
fi
EOF

chmod +x "$WORKSPACE_DIR/configure_api.sh"

# Debug/Info script
cat > "$WORKSPACE_DIR/info_otw.sh" << 'EOF'
#!/bin/bash

echo "🔍 OpenTuneWeaver System Information"
echo "===================================="
echo ""
echo "📁 Installation:"
echo "  OTW Directory: ${OTW_DIR:-Not set}"
echo "  Workspace: ${WORKSPACE_DIR:-Not set}"
echo ""
echo "🔌 API Configuration:"
echo "  API URL: ${API_BASE_URL:-Not configured}"
echo "  Model: ${MODEL_NAME:-Not configured}"
echo ""
echo "🖥️ System:"
echo "  Python: $(python3 --version)"
echo "  CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Not available')"
echo ""
echo "📦 Key Python Packages:"
python3 -c "
import pkg_resources
packages = ['torch', 'transformers', 'gradio', 'peft', 'datasets']
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        print(f'  {pkg}: {version}')
    except:
        print(f'  {pkg}: Not installed')
"
echo ""
echo "📊 Directory Structure:"
if [ -d "$OTW_DIR" ]; then
    echo "  ✅ Pipeline modules present"
    ls -d "$OTW_DIR"/pipeline/modules/*/ 2>/dev/null | wc -l | xargs echo "  Modules found:"
else
    echo "  ❌ OTW directory not found"
fi
EOF

chmod +x "$WORKSPACE_DIR/info_otw.sh"

log "✅ Startup scripts created"

# ============================================
# SCHRITT 11: Final Tests
# ============================================

log "${BLUE}🧪 Running final tests...${NC}"

cd "$OTW_DIR"

# Test Python imports
python3 -c "
import sys
try:
    import torch
    import transformers
    import gradio
    import datasets
    import peft
    print('✅ Core packages imported successfully')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  Transformers: {transformers.__version__}')
    print(f'  Gradio: {gradio.__version__}')
    if torch.cuda.is_available():
        print(f'  CUDA: Available - {torch.cuda.get_device_name(0)}')
    else:
        print(f'  CUDA: Not available (CPU mode)')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

# Test app.py
if [ -f "app.py" ]; then
    python3 -m py_compile app.py 2>/dev/null && log "✅ app.py syntax check passed" || warning "app.py syntax check failed"
fi

# ============================================
# SCHRITT 12: Installation Summary
# ============================================

log "${GREEN}========================================${NC}"
log "${GREEN}✅ Installation Complete!${NC}"
log "${GREEN}========================================${NC}"
echo ""
echo "${MAGENTA}🎯 Quick Start Guide:${NC}"
echo ""
echo "1️⃣  Configure API (if not already done):"
echo "    ${CYAN}./configure_api.sh${NC}"
echo ""
echo "2️⃣  Start OpenTuneWeaver:"
echo "    ${CYAN}./start_otw.sh${NC}"
echo ""
echo "3️⃣  Access the UI:"
echo "    ${CYAN}http://localhost:8080${NC}"
echo ""
echo "${MAGENTA}📋 Other Commands:${NC}"
echo "  System info:  ${CYAN}./info_otw.sh${NC}"
echo "  Logs:         ${CYAN}tail -f $OTW_DIR/logs/*.log${NC}"
echo ""
echo "${MAGENTA}🔧 Configuration:${NC}"
echo "  API URL:      ${CYAN}${API_BASE_URL}${NC}"
echo "  Model:        ${CYAN}${MODEL_NAME}${NC}"
echo "  Config file:  ${CYAN}$OTW_DIR/.env${NC}"
echo ""
echo "${YELLOW}💡 Tips:${NC}"
echo "  - Edit .env file to change API settings"
echo "  - Use any OpenAI-compatible API (Ollama, vLLM, OpenAI, etc.)"
echo "  - Check logs directory for debugging"
echo ""

# Ask if user wants to configure now
if [ -z "$API_BASE_URL" ] || [ "$API_BASE_URL" = "http://localhost:11434/v1" ]; then
    read -p "📡 Configure API connection now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        "$WORKSPACE_DIR/configure_api.sh"
    fi
fi

# Ask if user wants to start now
read -p "🚀 Start OpenTuneWeaver now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Starting OpenTuneWeaver..."
    "$WORKSPACE_DIR/start_otw.sh"
fi