#!/bin/bash

# ============================================
# OpenTuneWeaver RunPod Setup (FIXED VERSION)
# Version: 2.6 - Modell-Konsistenz-Fix
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
# SCHRITT 1: System Info und Vorbereitung
# ============================================

log "${BLUE}========================================${NC}"
log "${BLUE}🚀 OpenTuneWeaver RunPod Installation v2.6${NC}"
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
# SCHRITT 5: Python Environment Setup
# ============================================

log "${BLUE}🐍 Setting up Python environment...${NC}"

# Upgrade pip first
python3 -m pip install --upgrade pip setuptools wheel

# Install requirements from repository
if [ -f "requirements.txt" ]; then
    log "Installing packages from repository requirements.txt..."
    pip3 install -r requirements.txt
    log "✅ Repository requirements installed"
else
    error "requirements.txt not found in repository!"
fi

# Install additional packages that might be needed
log "Installing additional ML packages..."
pip3 install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install xformers with proper version constraints
if [ "$CUDA_AVAILABLE" = true ]; then
    pip3 install --no-deps "xformers<0.0.27"
    log "✅ CUDA-specific packages installed"
fi

log "✅ Python environment setup complete"

# ============================================
# SCHRITT 6: Create HuggingFace Cache Directory
# ============================================

log "${BLUE}🗂️ Setting up model cache directories...${NC}"

# Create consistent cache directories
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache/transformers"
export HF_DATASETS_CACHE="/workspace/hf_cache/datasets"

mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE

log "✅ Cache directories created:"
log "  HF_HOME: $HF_HOME"
log "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"

# ============================================
# SCHRITT 7: Safe Path Fixes (KORRIGIERT)
# ============================================

log "${BLUE}🔧 Safe path fixes...${NC}"

# Backup original files
if [ -f "ui/app.py" ]; then
    cp ui/app.py ui/app.py.backup
    log "Created backup of app.py"
fi

# Create symbolic link for pipeline access from UI
cd ui
if [ ! -L "pipeline" ]; then
    ln -sf ../pipeline pipeline
    log "✅ Created symbolic link ui/pipeline -> ../pipeline"
fi
cd ..

# Safe path corrections in app.py (nur einfache Ersetzungen)
if [ -f "ui/app.py" ]; then
    # Nur einfache Pfad-Ersetzungen - keine komplexen sed-Operationen
    sed -i 's|"../pipeline/|"pipeline/|g' ui/app.py
    sed -i 's|Path("../pipeline")|Path("pipeline")|g' ui/app.py
    log "✅ Applied safe path corrections to ui/app.py"
fi

# Safe corrections for other files
if [ -f "pipeline/run_pipeline.py" ]; then
    sed -i 's|Path("../|Path("|g' pipeline/run_pipeline.py
    log "✅ Fixed pipeline/run_pipeline.py paths"
fi

if [ -f "pipeline/config_loader.py" ]; then
    sed -i 's|Path.cwd().parent.parent.parent|Path.cwd().parent|g' pipeline/config_loader.py
    log "✅ Fixed pipeline/config_loader.py paths"
fi

# ============================================
# SCHRITT 8: Build llama.cpp (CPU-ONLY)
# ============================================

log "${BLUE}🔨 Building llama.cpp (CPU-Only)...${NC}"

# Navigate to finetuning directory
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

# CPU-Only Build (proven to work)
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

# Start Ollama in background with robust startup
log "Starting Ollama service..."
ollama serve > /workspace/ollama.log 2>&1 &
OLLAMA_PID=$!
echo $OLLAMA_PID > /workspace/ollama.pid

# Wait for Ollama to start with proper verification
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

log "${BLUE}📥 Downloading Ollama models (this will take a while)...${NC}"

# Download the desired smaller model for consistency
log "Downloading llama3.2:3b..."
if ollama pull llama3.2:3b; then
    log "✅ llama3.2:3b downloaded successfully"
    export OLLAMA_MODEL="llama3.2:3b"
else
    error "Failed to download llama3.2:3b"
fi

# Verify model is available
log "Available models:"
ollama list

# ============================================
# SCHRITT 11: Create FIXED Pipeline Configuration
# ============================================

log "${BLUE}📝 Creating FIXED pipeline configuration...${NC}"

# IMPORTANT: Use consistent model names!
FINETUNING_BASE_MODEL="unsloth/llama-3.2-3b-instruct"
FINETUNING_OUTPUT_NAME="OTW-Model-RunPod"

cat > /workspace/OpenTuneWeaver/pipeline/pipeline_config.json << EOF
{
  "version": "2.6-runpod-fixed",
  "created": "$(date -Iseconds)",
  "tokens": {
    "hf_token": "",
    "hf_write_token": ""
  },
  "api_configs": {
    "01_convert": {
      "use_openai_api": true,
      "openai_base_url": "http://localhost:11434/v1",
      "openai_api_key": "ollama",
      "openai_model_name": "${OLLAMA_MODEL}",
      "temperature": 0.1
    },
    "02_genwiki": {
      "use_openai_api": true,
      "openai_base_url": "http://localhost:11434/v1",
      "openai_api_key": "ollama",
      "openai_model_name": "${OLLAMA_MODEL}",
      "temperature": 0.3
    },
    "03_instructQA": {
      "use_openai_api": true,
      "openai_base_url": "http://localhost:11434/v1",
      "openai_api_key": "ollama",
      "openai_model_name": "${OLLAMA_MODEL}",
      "temperature": 0.7
    },
    "05_bmcreator": {
      "use_openai_api": true,
      "openai_base_url": "http://localhost:11434/v1",
      "openai_api_key": "ollama",
      "openai_model_name": "${OLLAMA_MODEL}",
      "temperature": 0.5
    }
  },
  "finetuning": {
    "model_name": "${FINETUNING_OUTPUT_NAME}",
    "base_model": "${FINETUNING_BASE_MODEL}",
    "hf_repo_id": "user/${FINETUNING_OUTPUT_NAME}",
    "dataset_path": "INPUT/dataset.json",
    "chat_template": "llama",
    "custom_model_dir": "CustomModel",
    "output_dir": "/workspace/OpenTuneWeaver/pipeline/modules/06_finetuning/CustomModel",
    "cache_dir": "${TRANSFORMERS_CACHE}",
    "max_seq_length": 2048,
    "load_in_4bit": true,
    "full_finetuning": false,
    "lora_r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 100,
    "num_train_epochs": 3,
    "learning_rate": 5e-5,
    "save_lora": true,
    "save_merged": true,
    "save_gguf": false
  },
  "benchmark": {
    "mode": "comparison",
    "base_model": "${FINETUNING_BASE_MODEL}",
    "finetuned_model": "/workspace/OpenTuneWeaver/pipeline/modules/06_finetuning/CustomModel",
    "model_cache_dir": "${TRANSFORMERS_CACHE}",
    "evaluator": {
      "type": "api",
      "api_base_url": "http://localhost:11434/v1",
      "api_key": "ollama",
      "model": "${OLLAMA_MODEL}"
    }
  },
  "pipeline": {
    "auto_cleanup": false,
    "verbose": true,
    "continue_on_error": true
  },
  "environment": {
    "HF_HOME": "${HF_HOME}",
    "TRANSFORMERS_CACHE": "${TRANSFORMERS_CACHE}",
    "HF_DATASETS_CACHE": "${HF_DATASETS_CACHE}"
  }
}
EOF

log "✅ FIXED Configuration created with consistent model names:"
log "  Base Model: ${FINETUNING_BASE_MODEL}"
log "  Output Model: ${FINETUNING_OUTPUT_NAME}"
log "  Ollama Model: ${OLLAMA_MODEL}"

# ============================================
# SCHRITT 12: Create Directory Structure
# ============================================

log "${BLUE}📁 Creating directory structure...${NC}"

cd /workspace/OpenTuneWeaver

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

# Create model storage directories
mkdir -p /workspace/OpenTuneWeaver/pipeline/modules/06_finetuning/CustomModel
mkdir -p $TRANSFORMERS_CACHE

log "✅ Directory structure created"

# ============================================
# SCHRITT 13: Create Environment Script
# ============================================

log "${BLUE}📝 Creating environment script...${NC}"

cat > /workspace/set_env.sh << 'EOF'
#!/bin/bash

# Set environment variables for OpenTuneWeaver
export HF_HOME="/workspace/hf_cache"
export TRANSFORMERS_CACHE="/workspace/hf_cache/transformers"
export HF_DATASETS_CACHE="/workspace/hf_cache/datasets"
export CUDA_VISIBLE_DEVICES="0"

echo "✅ Environment variables set:"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
EOF

chmod +x /workspace/set_env.sh

# ============================================
# SCHRITT 14: Create Enhanced Startup Scripts
# ============================================

log "${BLUE}📝 Creating enhanced startup scripts...${NC}"

# Enhanced startup script with environment setup
cat > /workspace/start_otw.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting OpenTuneWeaver with FIXED configuration..."

# Set environment variables
source /workspace/set_env.sh

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

# Wait for Ollama to be ready
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

# Verify UI structure
echo "📁 Verifying UI structure..."
cd /workspace/OpenTuneWeaver/ui
if [ ! -L "pipeline" ]; then
    echo "🔧 Creating pipeline symlink..."
    ln -sf ../pipeline pipeline
    echo "✅ Pipeline symlink created"
fi

# Verify configuration
echo "📋 Verifying configuration..."
if [ -f "/workspace/OpenTuneWeaver/pipeline/pipeline_config.json" ]; then
    echo "✅ Configuration file found"
else
    echo "❌ Configuration file missing!"
    exit 1
fi

# Start OpenTuneWeaver UI
echo "Starting OpenTuneWeaver UI on port 8080..."
python3 app.py --server_name 0.0.0.0 --server_port 8080
EOF

chmod +x /workspace/start_otw.sh

# Enhanced debug script
cat > /workspace/debug_otw.sh << 'EOF'
#!/bin/bash

echo "🔍 OpenTuneWeaver Debug Information v2.6"
echo "========================================"

# Set environment
source /workspace/set_env.sh

echo "🌍 Environment Variables:"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"

echo -e "\n📁 Directory structure:"
ls -la /workspace/OpenTuneWeaver/ | head -10

echo -e "\nUI directory:"
ls -la /workspace/OpenTuneWeaver/ui/ 2>/dev/null || echo "UI directory not found"

echo -e "\nPipeline directory:"
ls -la /workspace/OpenTuneWeaver/pipeline/ | head -5 2>/dev/null || echo "Pipeline directory not found"

echo -e "\nFinetuning directory:"
ls -la /workspace/OpenTuneWeaver/pipeline/modules/06_finetuning/ 2>/dev/null || echo "Finetuning directory not found"

echo -e "\nCustomModel directory:"
ls -la /workspace/OpenTuneWeaver/pipeline/modules/06_finetuning/CustomModel/ 2>/dev/null || echo "CustomModel directory not found"

echo -e "\nCache directories:"
ls -la /workspace/hf_cache/ 2>/dev/null || echo "Cache directory not found"
ls -la /workspace/hf_cache/transformers/ 2>/dev/null || echo "Transformers cache not found"

echo -e "\nSymbolic links:"
find /workspace/OpenTuneWeaver/ui/ -type l -ls 2>/dev/null || echo "No symbolic links found"

echo -e "\nConfiguration file:"
if [ -f "/workspace/OpenTuneWeaver/pipeline/pipeline_config.json" ]; then
    echo "✅ Configuration file exists"
    echo "Base model: $(grep -o '"base_model": "[^"]*"' /workspace/OpenTuneWeaver/pipeline/pipeline_config.json | cut -d'"' -f4)"
    echo "Output dir: $(grep -o '"output_dir": "[^"]*"' /workspace/OpenTuneWeaver/pipeline/pipeline_config.json | cut -d'"' -f4)"
else
    echo "❌ Configuration file missing"
fi

echo -e "\nApp.py syntax check:"
cd /workspace/OpenTuneWeaver/ui 2>/dev/null && python3 -m py_compile app.py 2>/dev/null && echo "✅ app.py syntax OK" || echo "❌ app.py syntax error"

echo -e "\nOllama status:"
if curl -s http://localhost:11434/api/tags 2>/dev/null; then
    echo "✅ Ollama responding"
    ollama list
else
    echo "❌ Ollama not responding"
fi

echo -e "\nPython packages:"
pip3 list | grep -E "(torch|transformers|gradio|unsloth)" | head -5

echo -e "\nGPU Status:"
nvidia-smi 2>/dev/null || echo "No GPU or nvidia-smi not available"
EOF

chmod +x /workspace/debug_otw.sh

log "✅ Enhanced startup scripts created"

# ============================================
# SCHRITT 15: Final Installation Test
# ============================================

log "${BLUE}🧪 Final installation test...${NC}"

cd /workspace/OpenTuneWeaver

# Set environment for test
source /workspace/set_env.sh

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
        print(f'  CUDA: Yes - {torch.cuda.get_device_name(0)}')
    else:
        print(f'  CUDA: No (CPU-only mode)')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

# Test app.py syntax
cd ui 2>/dev/null && python3 -m py_compile app.py 2>/dev/null && log "✅ app.py syntax verification passed" || warning "app.py syntax verification failed"

# Test Ollama connection
if curl -s http://localhost:11434/api/tags > /dev/null; then
    log "✅ Ollama connection test passed"
else
    warning "Ollama connection test failed"
fi

# Test configuration
if [ -f "/workspace/OpenTuneWeaver/pipeline/pipeline_config.json" ]; then
    log "✅ Configuration file created successfully"
else
    error "Configuration file missing!"
fi

# ============================================
# SCHRITT 16: Installation Complete
# ============================================

log "${GREEN}========================================${NC}"
log "${GREEN}✅ Installation Complete! (FIXED Version)${NC}"
log "${GREEN}========================================${NC}"

echo ""
echo "🔧 FIXED Issues:"
echo "  ✅ Consistent model names in configuration"
echo "  ✅ Proper cache directory setup"
echo "  ✅ Fixed base model: ${FINETUNING_BASE_MODEL}"
echo "  ✅ Fixed output model: ${FINETUNING_OUTPUT_NAME}"
echo "  ✅ Enhanced environment variable management"
echo ""
echo "📋 Quick Start Commands:"
echo "  Start OpenTuneWeaver:  /workspace/start_otw.sh"
echo "  Debug information:     /workspace/debug_otw.sh"
echo "  Set environment:       source /workspace/set_env.sh"
echo ""
echo "🌐 Access URLs:"
echo "  OpenTuneWeaver UI:     http://[POD-IP]:8080"
echo "  Ollama API:           http://[POD-IP]:11434"
echo ""
echo "🤖 Models:"
echo "  Base Model: ${FINETUNING_BASE_MODEL}"
echo "  Ollama Model: ${OLLAMA_MODEL}"
echo "  Output Model: ${FINETUNING_OUTPUT_NAME}"
echo ""

# Optional: Auto-start
read -p "🚀 Start OpenTuneWeaver now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Starting OpenTuneWeaver..."
    /workspace/start_otw.sh
fi