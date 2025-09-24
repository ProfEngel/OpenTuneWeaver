#!/bin/bash

# ============================================
# OpenTuneWeaver RunPod Setup with venv
# Version: 4.0-venv
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

# Virtual Environment Pfad
VENV_PATH="/workspace/otw_venv"
VENV_PYTHON="${VENV_PATH}/bin/python"
VENV_PIP="${VENV_PATH}/bin/pip"

# ============================================
# SCHRITT 1: System Info und Vorbereitung
# ============================================

log "${BLUE}========================================${NC}"
log "${BLUE}🚀 OpenTuneWeaver venv Setup v4.0${NC}"
log "${BLUE}========================================${NC}"

# System Info
log "📊 System Information:"
echo "  Hostname: $(hostname)"
echo "  CPU: $(nproc) cores"
echo "  RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "  Python: $(python3 --version)"
echo "  Current Dir: $(pwd)"

# ============================================
# SCHRITT 2: System Dependencies
# ============================================

log "${BLUE}📦 Installing System Dependencies...${NC}"
apt-get update
apt-get upgrade -y
apt-get install -y \
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
    supervisor \
    htop \
    nvtop \
    tmux \
    nano \
    unzip

log "✅ System dependencies installed"

# ============================================
# SCHRITT 3: NVIDIA/CUDA Verification
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
# SCHRITT 4: Clone Repository
# ============================================

log "${BLUE}📥 Cloning OpenTuneWeaver Repository...${NC}"
cd /workspace

# Remove if exists
if [ -d "OpenTuneWeaver" ]; then
    warning "OpenTuneWeaver directory exists, removing..."
    rm -rf OpenTuneWeaver
fi

# Clone repository
git clone https://github.com/ProfEngel/OpenTuneWeaver.git
cd OpenTuneWeaver

log "✅ Repository cloned successfully"

# ============================================
# SCHRITT 5: Create Python Virtual Environment
# ============================================

log "${BLUE}🐍 Creating Python Virtual Environment...${NC}"

# Remove old venv if exists
if [ -d "$VENV_PATH" ]; then
    warning "Old venv found, removing..."
    rm -rf "$VENV_PATH"
fi

# Create new venv
python3 -m venv "$VENV_PATH"

# Activate venv
source "${VENV_PATH}/bin/activate"

# Verify venv is active
which python
python --version

log "✅ Virtual environment created at $VENV_PATH"

# ============================================
# SCHRITT 6: Install Python Packages in venv
# ============================================

log "${BLUE}📦 Installing Python packages in venv...${NC}"

# Upgrade pip, setuptools, wheel in venv
${VENV_PIP} install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support if available
if [ "$CUDA_AVAILABLE" = true ]; then
    log "Installing PyTorch with CUDA support..."
    ${VENV_PIP} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    log "Installing PyTorch (CPU only)..."
    ${VENV_PIP} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install requirements from repository
if [ -f "requirements.txt" ]; then
    log "Installing packages from requirements.txt..."
    ${VENV_PIP} install -r requirements.txt
    log "✅ Repository requirements installed"
else
    error "requirements.txt not found in repository!"
fi

# Install unsloth
log "Installing unsloth..."
${VENV_PIP} install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install unsloth_zoo
log "Installing unsloth_zoo..."
${VENV_PIP} install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth-zoo.git

# Install xformers if CUDA available
if [ "$CUDA_AVAILABLE" = true ]; then
    ${VENV_PIP} install --no-deps "xformers<0.0.27"
    log "✅ CUDA-specific packages installed"
fi

log "✅ Python packages installed in venv"

# ============================================
# SCHRITT 7: Setup Directory Structure
# ============================================

log "${BLUE}🔧 Setting up directory structure...${NC}"

# Move app.py to root if needed
if [ ! -f "app.py" ]; then
    if [ -f "ui/app_new.py" ]; then
        log "Moving app_new.py to root as app.py..."
        cp ui/app_new.py app.py
        log "✅ app.py placed in root directory"
    elif [ -f "ui/app.py" ]; then
        log "Moving ui/app.py to root..."
        cp ui/app.py app.py
        log "✅ app.py placed in root directory"
    fi
fi

chmod +x app.py 2>/dev/null || true

log "✅ Directory structure ready"

# ============================================
# SCHRITT 8: Build llama.cpp (CPU-ONLY)
# ============================================

log "${BLUE}🔨 Building llama.cpp (CPU-Only)...${NC}"

cd /workspace/OpenTuneWeaver/pipeline/modules/06_finetuning

# Clone llama.cpp if not exists
if [ ! -d "llama.cpp" ]; then
    log "Cloning llama.cpp..."
    git clone --recursive https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp

# Clean previous builds
rm -rf build
mkdir build
cd build

# CPU-Only Build
log "Building llama.cpp CPU-only version..."
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Verify build
if [ -f "bin/llama-cli" ] || [ -f "bin/main" ]; then
    log "✅ llama.cpp built successfully (CPU-only)"
else
    error "llama.cpp build failed!"
fi

cd /workspace/OpenTuneWeaver

# ============================================
# SCHRITT 9: Install and Configure Ollama
# ============================================

log "${BLUE}🦙 Installing Ollama...${NC}"

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Kill any existing Ollama processes
pkill ollama 2>/dev/null || true
sleep 2

# Start Ollama in background
log "Starting Ollama service..."
ollama serve > /workspace/ollama.log 2>&1 &
OLLAMA_PID=$!
echo $OLLAMA_PID > /workspace/ollama.pid

# Wait for Ollama to start
log "Waiting for Ollama to become ready..."
for i in {1..60}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log "✅ Ollama is running successfully on port 11434"
        break
    else
        if [ $i -eq 60 ]; then
            error "Ollama failed to start after 60 attempts"
        fi
        echo "  Waiting for Ollama... ($i/60)"
        sleep 2
    fi
done

# ============================================
# SCHRITT 10: Download Ollama Models
# ============================================

log "${BLUE}📥 Downloading Ollama models...${NC}"

# Download model
log "Downloading gemma3:12b-it-qat (this may take 10-15 minutes)..."
if ollama pull gemma3:12b-it-qat; then
    log "✅ gemma3:12b-it-qat downloaded successfully"
    export OLLAMA_MODEL="gemma3:12b-it-qat"
else
    warning "Failed to download gemma3:12b-it-qat, falling back to gemma3:4b-it-qat"
    ollama pull gemma3:4b-it-qat
    export OLLAMA_MODEL="gemma3:4b-it-qat"
fi

# Verify model
log "Available models:"
ollama list

# ============================================
# SCHRITT 11: Create Pipeline Configuration
# ============================================

log "${BLUE}📝 Creating pipeline configuration...${NC}"

cat > /workspace/OpenTuneWeaver/pipeline/pipeline_config.json << EOF
{
  "version": "4.0-venv",
  "created": "$(date -Iseconds)",
  "venv_path": "${VENV_PATH}",
  "tokens": {
    "hf_token": "",
    "hf_write_token": ""
  },
  "api_configs": {
    "01_convert": {
      "use_openai_api": true,
      "openai_base_url": "http://localhost:11434/v1",
      "openai_api_key": "ollama",
      "openai_model_name": "${OLLAMA_MODEL:-gemma3:12b-it-qat}",
      "temperature": 0.1
    },
    "02_genwiki": {
      "use_openai_api": true,
      "openai_base_url": "http://localhost:11434/v1",
      "openai_api_key": "ollama",
      "openai_model_name": "${OLLAMA_MODEL:-gemma3:12b-it-qat}",
      "temperature": 0.3
    },
    "03_instructQA": {
      "use_openai_api": true,
      "openai_base_url": "http://localhost:11434/v1",
      "openai_api_key": "ollama",
      "openai_model_name": "${OLLAMA_MODEL:-gemma3:12b-it-qat}",
      "temperature": 0.7
    },
    "05_bmcreator": {
      "use_openai_api": true,
      "openai_base_url": "http://localhost:11434/v1",
      "openai_api_key": "ollama",
      "openai_model_name": "${OLLAMA_MODEL:-gemma3:12b-it-qat}",
      "temperature": 0.5
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
    "save_gguf": false,
    "upload_to_hf": false,
    "gguf_quantizations": ["q8_0"],
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
      "type": "unknown",
      "load_in_4bit": false,
      "max_seq_length": 2048,
      "base_model": null
    },
    "evaluator": {
      "type": "api",
      "api_base_url": "http://localhost:11434/v1",
      "api_key": "ollama",
      "model": "${OLLAMA_MODEL:-gemma3:12b-it-qat}"
    },
    "questions_file": "BENCHMARKFRAGEN/benchmark_fragen_complete.json",
    "max_new_tokens": 256,
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
  },
  "pipeline": {
    "auto_cleanup": false,
    "verbose": true,
    "continue_on_error": true,
    "save_metrics": true
  }
}
EOF

log "✅ Configuration created"

# ============================================
# SCHRITT 12: Create Directory Structure
# ============================================

log "${BLUE}📁 Creating directory structure...${NC}"

cd /workspace/OpenTuneWeaver

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
# SCHRITT 13: Create Startup Scripts with venv
# ============================================

log "${BLUE}📝 Creating venv-aware startup scripts...${NC}"

# Main startup script
cat > /workspace/start_otw.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting OpenTuneWeaver with venv..."

# Virtual Environment Path
VENV_PATH="/workspace/otw_venv"

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run setup script first!"
    exit 1
fi

# Function to check if Ollama is responding
check_ollama() {
    curl -s http://localhost:11434/api/tags > /dev/null 2>&1
}

# Kill existing Ollama processes
pkill ollama 2>/dev/null || true
sleep 3

# Start Ollama
echo "Starting Ollama..."
ollama serve > /workspace/ollama.log 2>&1 &
OLLAMA_PID=$!
echo $OLLAMA_PID > /workspace/ollama.pid

# Wait for Ollama
echo "Waiting for Ollama to become ready..."
for i in {1..30}; do
    if check_ollama; then
        echo "✅ Ollama is running and responding"
        break
    else
        if [ $i -eq 30 ]; then
            echo "❌ Ollama failed to start properly"
            echo "Check logs: tail -f /workspace/ollama.log"
            exit 1
        fi
        echo "Waiting... ($i/30)"
        sleep 2
    fi
done

# Activate venv and start OpenTuneWeaver
echo "Activating virtual environment..."
source "${VENV_PATH}/bin/activate"

echo "Starting OpenTuneWeaver UI on port 8080..."
cd /workspace/OpenTuneWeaver
python app.py --server_name 0.0.0.0 --server_port 8080
EOF

chmod +x /workspace/start_otw.sh

# Debug script
cat > /workspace/debug_otw.sh << 'EOF'
#!/bin/bash

VENV_PATH="/workspace/otw_venv"

echo "🔍 OpenTuneWeaver Debug Information v4.0-venv"
echo "=============================================="

echo "📁 Virtual Environment:"
if [ -d "$VENV_PATH" ]; then
    echo "✅ venv exists at $VENV_PATH"
    echo "  Python: $($VENV_PATH/bin/python --version)"
    echo ""
    echo "📦 Installed packages (key ones):"
    $VENV_PATH/bin/pip list | grep -E "(torch|transformers|gradio|unsloth)" | head -10
else
    echo "❌ venv not found at $VENV_PATH"
fi

echo ""
echo "📁 OpenTuneWeaver structure:"
ls -la /workspace/OpenTuneWeaver/ | head -10

echo ""
echo "App location:"
ls -la /workspace/OpenTuneWeaver/app.py 2>/dev/null || echo "app.py not found"

echo ""
echo "🦙 Ollama status:"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama responding"
    ollama list
else
    echo "❌ Ollama not responding"
fi
EOF

chmod +x /workspace/debug_otw.sh

# Activate venv script (for manual work)
cat > /workspace/activate_otw_venv.sh << 'EOF'
#!/bin/bash
echo "Activating OpenTuneWeaver virtual environment..."
source /workspace/otw_venv/bin/activate
echo "✅ Virtual environment activated"
echo "Python: $(which python)"
echo "Version: $(python --version)"
EOF

chmod +x /workspace/activate_otw_venv.sh

log "✅ Startup scripts created"

# ============================================
# SCHRITT 14: Test Installation
# ============================================

log "${BLUE}🧪 Testing installation...${NC}"

cd /workspace/OpenTuneWeaver

# Test imports using venv Python
${VENV_PYTHON} -c "
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
        print(f'  CUDA: Yes - {torch.cuda.get_device_name(0)}')
    else:
        print(f'  CUDA: No (CPU-only mode)')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

# Test app.py
if [ -f "app.py" ]; then
    ${VENV_PYTHON} -m py_compile app.py 2>/dev/null && log "✅ app.py syntax verification passed" || warning "app.py syntax verification failed"
else
    warning "app.py not found"
fi

# Test Ollama
if curl -s http://localhost:11434/api/tags > /dev/null; then
    log "✅ Ollama connection test passed"
else
    warning "Ollama connection test failed"
fi

# ============================================
# SCHRITT 15: Installation Complete
# ============================================

# Deactivate venv for clean state
deactivate 2>/dev/null || true

log "${GREEN}========================================${NC}"
log "${GREEN}✅ Installation Complete! (v4.0-venv)${NC}"
log "${GREEN}========================================${NC}"

echo ""
echo "📁 Virtual Environment:"
echo "  Location: ${VENV_PATH}"
echo "  Python: ${VENV_PYTHON}"
echo ""
echo "📋 Quick Start Commands:"
echo "  Start OTW:            /workspace/start_otw.sh"
echo "  Debug info:           /workspace/debug_otw.sh"
echo "  Activate venv:        source /workspace/otw_venv/bin/activate"
echo "  Or use:               /workspace/activate_otw_venv.sh"
echo ""
echo "🌐 Access URLs:"
echo "  OpenTuneWeaver UI:    http://[POD-IP]:8080"
echo "  Ollama API:          http://[POD-IP]:11434"
echo ""
echo "🤖 Model: ${OLLAMA_MODEL:-gemma3:12b-it-qat}"
echo ""

# Auto-start prompt
read -p "🚀 Start OpenTuneWeaver now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Starting OpenTuneWeaver..."
    /workspace/start_otw.sh
fi