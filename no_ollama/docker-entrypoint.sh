#!/bin/bash
set -e

echo "🚀 Starting OpenTuneWeaver..."
echo "📍 Working directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "🔥 PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "🎮 CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Setze HuggingFace Cache
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# Erstelle Cache-Verzeichnisse
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE

# Optional: HuggingFace Login (wenn Token vorhanden)
if [ ! -z "$HF_TOKEN" ]; then
    echo "🔑 Logging into HuggingFace..."
    huggingface-cli login --token $HF_TOKEN
fi

# Starte die Gradio App
echo "🌐 Starting Gradio UI on port 8080..."
cd /workspace/OpenTuneWeaver/ui
python app.py

# Halte Container am Leben falls App crashed
tail -f /dev/null