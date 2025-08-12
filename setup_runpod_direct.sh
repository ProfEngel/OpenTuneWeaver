#!/bin/bash

# ============================================
# OpenTuneWeaver RunPod Setup (CPU-Build Fix)
# Version: 2.2
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
log "${BLUE}🚀 OpenTuneWeaver RunPod Installation v2.2${NC}"
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
# SCHRITT 6: Fix Path Issues in Python Files
# ============================================

log "${BLUE}🔧 Fixing path issues...${NC}"

# Fix app.py path issue (../pipeline -> pipeline)
if [ -f "ui/app.py" ]; then
    sed -i 's|"../pipeline/|"pipeline/|g' ui/app.py
    sed -i 's|Path("../pipeline")|Path("pipeline")|g' ui/app.py
    log "✅ Fixed ui/app.py paths"
fi

# Fix run_pipeline.py paths if needed
if [ -f "pipeline/run_pipeline.py" ]; then
    sed -i 's|Path("../|Path("|g' pipeline/run_pipeline.py
    log "✅ Fixed pipeline/run_pipeline.py paths"
fi

# Fix config_loader.py paths
if [ -f "pipeline/config_loader.py" ]; then
    sed -i 's|Path.cwd().parent.parent.parent|Path.cwd().parent.parent|g' pipeline/config_loader.py
    log "✅ Fixed pipeline/config_loader.py paths"
fi

# ============================================
# SCHRITT 7: Build llama.cpp (CPU-ONLY)
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
# SCHRITT 8: Install and Configure Ollama
# ============================================

log "${BLUE}🦙 Installing Ollama...${NC}"

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama in background
log "Starting Ollama service..."
nohup ollama serve > /workspace/ollama.log 2>&1 &
OLLAMA_PID=$!
echo $OLLAMA_PID > /workspace/ollama.pid

# Wait for Ollama to start
sleep 10

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    log "✅ Ollama is running successfully"
else
    warning "Ollama might not be running properly, check /workspace/ollama.log"
fi

# ============================================
# SCHRITT 9: Download Ollama Models
# ============================================

log "${BLUE}📥 Downloading Ollama models...${NC}"

# Download a lightweight model for testing
ollama pull llama3.2:3b || warning "Failed to pull model - will retry later"

log "✅ Models downloaded"

# ============================================
# SCHRITT 10: Create Pipeline Configuration
# ============================================

log "${BLUE}📝 Creating pipeline configuration...${NC}"

cat > /workspace/OpenTuneWeaver/pipeline/pipeline_config.json << 'EOF'
{
  "version": "2.2-runpod",
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
      "openai_model_name": "llama3.2:3b",
      "temperature": 0.1
    },
    "02_genwiki": {
      "use_openai_api": true,
      "openai_base_url": "http://localhost:11434/v1",
      "openai_api_key": "ollama",
      "openai_model_name": "llama3.2:3b",
      "temperature": 0.3
    },
    "03_instructQA": {
      "use_openai_api": true,
      "openai_base_url": "http://localhost:11434/v1",
      "openai_api_key": "ollama",
      "openai_model_name": "llama3.2:3b",
      "temperature": 0.7
    },
    "05_bmcreator": {
      "use_openai_api": true,
      "openai_base_url": "http://localhost:11434/v1",
      "openai_api_key": "ollama",
      "openai_model_name": "llama3.2:3b",
      "temperature": 0.5
    }
  },
  "finetuning": {
    "model_name": "OTW-Model",
    "base_model": "unsloth/gemma-2b-it",
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
    "evaluator": {
      "type": "api",
      "api_base_url": "http://localhost:11434/v1",
      "api_key": "ollama",
      "model": "llama3.2:3b"
    }
  },
  "pipeline": {
    "auto_cleanup": false,
    "verbose": true,
    "continue_on_error": true
  }
}
EOF

log "✅ Configuration created"

# ============================================
# SCHRITT 11: Create Directory Structure
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

log "✅ Directory structure created"

# ============================================
# SCHRITT 12: Create Startup Scripts
# ============================================

log "${BLUE}📝 Creating startup scripts...${NC}"

# Main startup script
cat > /workspace/start_otw.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting OpenTuneWeaver..."

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    nohup ollama serve > /workspace/ollama.log 2>&1 &
    sleep 10
fi

# Check Ollama status
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not responding"
    echo "Check logs: tail -f /workspace/ollama.log"
fi

# Start OpenTuneWeaver UI
cd /workspace/OpenTuneWeaver/ui
echo "Starting OpenTuneWeaver UI on port 8080..."
python3 app.py --server_name 0.0.0.0 --server_port 8080
EOF

chmod +x /workspace/start_otw.sh

# Debug script
cat > /workspace/debug_otw.sh << 'EOF'
#!/bin/bash

echo "🔍 OpenTuneWeaver Debug Information"
echo "=================================="

echo "Directory structure:"
ls -la /workspace/OpenTuneWeaver/

echo -e "\nPython packages:"
pip3 list | grep -E "(torch|transformers|gradio|unsloth)"

echo -e "\nOllama status:"
curl -s http://localhost:11434/api/tags 2>/dev/null || echo "Ollama not responding"

echo -e "\nGPU status:"
nvidia-smi 2>/dev/null || echo "No GPU available"

echo -e "\nProcess status:"
ps aux | grep -E "(ollama|python)" | head -10

echo -e "\nllama.cpp build status:"
ls -la /workspace/OpenTuneWeaver/pipeline/modules/06_finetuning/llama.cpp/build/bin/ 2>/dev/null || echo "llama.cpp not built"
EOF

chmod +x /workspace/debug_otw.sh

log "✅ Startup scripts created"

# ============================================
# SCHRITT 13: Installation Test
# ============================================

log "${BLUE}🧪 Testing installation...${NC}"

cd /workspace/OpenTuneWeaver

# Test Python imports
python3 -c "
import sys
try:
    import torch
    import transformers
    import gradio
    import datasets
    print('✅ Core packages imported successfully')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  Transformers: {transformers.__version__}')
    print(f'  Gradio: {gradio.__version__}')
    if torch.cuda.is_available():
        print(f'  CUDA available: Yes - {torch.cuda.get_device_name(0)}')
    else:
        print(f'  CUDA available: No (CPU-only mode)')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

# Test Ollama connection
if curl -s http://localhost:11434/api/tags > /dev/null; then
    log "✅ Ollama connection test passed"
else
    warning "Ollama connection test failed"
fi

# Test llama.cpp build
if [ -f "/workspace/OpenTuneWeaver/pipeline/modules/06_finetuning/llama.cpp/build/bin/llama-cli" ] || [ -f "/workspace/OpenTuneWeaver/pipeline/modules/06_finetuning/llama.cpp/build/bin/main" ]; then
    log "✅ llama.cpp build verification passed"
else
    warning "llama.cpp build verification failed"
fi

# ============================================
# SCHRITT 14: Final Setup Complete
# ============================================

log "${GREEN}========================================${NC}"
log "${GREEN}✅ Installation Complete!${NC}"
log "${GREEN}========================================${NC}"

echo ""
echo "📋 Quick Start Commands:"
echo "  Start OpenTuneWeaver:  /workspace/start_otw.sh"
echo "  Debug information:     /workspace/debug_otw.sh"
echo ""
echo "🌐 Access URLs:"
echo "  OpenTuneWeaver UI:     http://[POD-IP]:8080"
echo "  Ollama API:           http://[POD-IP]:11434"
echo ""
echo "📁 Important Paths:"
echo "  Project directory:    /workspace/OpenTuneWeaver"
echo "  Configuration:        /workspace/OpenTuneWeaver/pipeline/pipeline_config.json"
echo "  Logs:                /workspace/ollama.log"
echo "  llama.cpp binary:     /workspace/OpenTuneWeaver/pipeline/modules/06_finetuning/llama.cpp/build/bin/"
echo ""
echo "ℹ️  Build Info:"
echo "  llama.cpp: CPU-only build (stable and reliable)"
echo "  PyTorch: GPU-accelerated (if CUDA available)"
echo "  Ollama: GPU-accelerated (if CUDA available)"
echo ""

# Optional: Auto-start
read -p "🚀 Start OpenTuneWeaver now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Starting OpenTuneWeaver..."
    /workspace/start_otw.sh
fi
