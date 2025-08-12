#!/bin/bash

# ============================================
# OpenTuneWeaver RunPod Direct Installation
# Version: 1.0
# ============================================

set -e  # Exit on error

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging Funktion
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
log "${BLUE}🚀 OpenTuneWeaver RunPod Installation${NC}"
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
    nano

log "✅ System dependencies installed"

# ============================================
# SCHRITT 3: NVIDIA/CUDA Verification
# ============================================

log "${BLUE}🎮 Checking CUDA...${NC}"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    log "✅ CUDA is available"
else
    warning "No CUDA detected - CPU only mode"
fi

# ============================================
# SCHRITT 4: Clone Repository
# ============================================

log "${BLUE}📥 Cloning OpenTuneWeaver...${NC}"

cd /workspace

# Remove if exists
if [ -d "OpenTuneWeaver" ]; then
    warning "OpenTuneWeaver directory exists, removing..."
    rm -rf OpenTuneWeaver
fi

git clone https://github.com/ProfEngel/OpenTuneWeaver.git
cd OpenTuneWeaver

log "✅ Repository cloned"

# ============================================
# SCHRITT 5: Fix Path Issues in Python Files
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
    # Make sure it uses relative paths correctly
    sed -i 's|Path("../|Path("|g' pipeline/run_pipeline.py
    log "✅ Fixed pipeline/run_pipeline.py paths"
fi

# Fix config_loader.py paths
if [ -f "pipeline/config_loader.py" ]; then
    # Ensure it looks in the right places
    sed -i 's|Path.cwd().parent.parent.parent|Path.cwd().parent.parent|g' pipeline/config_loader.py
    log "✅ Fixed pipeline/config_loader.py paths"
fi

# ============================================
# SCHRITT 6: Python Dependencies
# ============================================

log "${BLUE}🐍 Installing Python dependencies...${NC}"

# Upgrade pip
python3 -m pip install --upgrade pip setuptools wheel

# Create a cleaned requirements.txt without version conflicts
cat > requirements_clean.txt << 'EOF'
# Core ML/AI
torch==2.2.0
torchvision
torchaudio
transformers==4.44.0
datasets==2.18.0
accelerate==0.30.1
peft==0.11.1
bitsandbytes==0.43.1
trl==0.9.6
safetensors
sentencepiece
tokenizers

# Gradio UI
gradio

# Document Processing
docling
pypdf
PyPDF2
fpdf
python-docx
python-pptx
openpyxl
beautifulsoup4
lxml

# Image Processing
Pillow
opencv-python

# API & Networking
openai
requests
aiohttp

# Utilities
numpy
pandas
matplotlib
seaborn
psutil
python-dotenv
huggingface-hub
scipy
scikit-learn
tqdm
pyyaml
protobuf

# GGUF
gguf
EOF

log "Installing Python packages..."
pip3 install -r requirements_clean.txt

# Install Unsloth
log "Installing Unsloth..."
pip3 install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip3 install --no-deps "xformers<0.0.27" peft accelerate bitsandbytes

log "✅ Python dependencies installed"

# ============================================
# SCHRITT 7: Build llama.cpp
# ============================================

log "${BLUE}🔨 Building llama.cpp...${NC}"

cd /workspace/OpenTuneWeaver/pipeline/modules/06_finetuning

# Clone llama.cpp if not exists
if [ ! -d "llama.cpp" ]; then
    git clone --recursive https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp

# Clean and build
rm -rf build
mkdir build
cd build

# Build with or without CUDA
if command -v nvidia-smi &> /dev/null; then
    log "Building with CUDA support..."
    cmake .. -DLLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=all
else
    log "Building CPU-only version..."
    cmake .. -DCMAKE_BUILD_TYPE=Release
fi

make -j$(nproc)

# Verify build
if [ -f "bin/llama-cli" ]; then
    log "✅ llama.cpp built successfully"
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
sleep 5

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    log "✅ Ollama is running"
else
    warning "Ollama might not be running properly"
fi

# ============================================
# SCHRITT 9: Create pipeline_config.json
# ============================================

log "${BLUE}📝 Creating pipeline configuration...${NC}"

cat > /workspace/OpenTuneWeaver/pipeline/pipeline_config.json << 'EOF'
{
  "version": "1.0-runpod",
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
# SCHRITT 10: Create necessary directories
# ============================================

log "${BLUE}📁 Creating directories...${NC}"

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

log "✅ Directories created"

# ============================================
# SCHRITT 11: Download a small Ollama model
# ============================================

log "${BLUE}📥 Downloading Ollama model (this may take a while)...${NC}"

# Use smaller model for testing (llama3.2:3b instead of gemma3:12b)
ollama pull llama3.2:3b || warning "Failed to pull model - will retry later"

# ============================================
# SCHRITT 12: Create startup script
# ============================================

log "${BLUE}📝 Creating startup script...${NC}"

cat > /workspace/start_otw.sh << 'EOF'
#!/bin/bash

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    nohup ollama serve > /workspace/ollama.log 2>&1 &
    sleep 5
fi

# Check Ollama
curl -s http://localhost:11434/api/tags || echo "Ollama not responding"

# Start OpenTuneWeaver
cd /workspace/OpenTuneWeaver
python3 ui/app.py --server_name 0.0.0.0 --server_port 8080
EOF

chmod +x /workspace/start_otw.sh

# ============================================
# SCHRITT 13: Test the installation
# ============================================

log "${BLUE}🧪 Testing installation...${NC}"

cd /workspace/OpenTuneWeaver

# Test Python imports
python3 -c "
import torch
import transformers
import gradio
print('✅ Core packages imported successfully')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# ============================================
# SCHRITT 14: Start the application
# ============================================

log "${GREEN}========================================${NC}"
log "${GREEN}✅ Installation Complete!${NC}"
log "${GREEN}========================================${NC}"

echo ""
echo "To start OpenTuneWeaver:"
echo "  /workspace/start_otw.sh"
echo ""
echo "Or manually:"
echo "  cd /workspace/OpenTuneWeaver"
echo "  python3 ui/app.py --server_name 0.0.0.0 --server_port 8080"
echo ""
echo "Access the UI at:"
echo "  http://[POD-IP]:8080"
echo ""

# Optional: Auto-start
read -p "Start OpenTuneWeaver now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    /workspace/start_otw.sh
fi