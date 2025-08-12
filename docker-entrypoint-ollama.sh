#!/bin/bash
set -e

echo "🚀 Starting OpenTuneWeaver with integrated Ollama..."
echo "📍 Working directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "🔥 PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "🎮 CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Setze HuggingFace Cache
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export OLLAMA_HOST=http://localhost:11434

# Erstelle Cache-Verzeichnisse
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE
mkdir -p /root/.ollama/models

# Optional: HuggingFace Login
if [ ! -z "$HF_TOKEN" ]; then
    echo "🔑 Logging into HuggingFace..."
    huggingface-cli login --token $HF_TOKEN --add-to-git-credential
fi

# Starte Ollama im Hintergrund
echo "🦙 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Warte bis Ollama bereit ist
echo "⏳ Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    echo "   Attempt $i/30..."
    sleep 2
done

# Prüfe ob Modell bereits vorhanden ist
echo "🔍 Checking for gemma3:12b model..."
if ! ollama list | grep -q "gemma3:12b"; then
    echo "📥 Pulling gemma3:12b model (this may take a while)..."
    ollama pull gemma3:12b
    echo "✅ Model gemma3:12b downloaded successfully!"
else
    echo "✅ Model gemma3:12b already available!"
fi

# Optional: Weitere Modelle
# ollama pull llama3.2:3b
# ollama pull mistral:7b

# Zeige verfügbare Modelle
echo "📋 Available Ollama models:"
ollama list

# Starte die Gradio App
echo "🌐 Starting Gradio UI on port 8080..."
cd /workspace/OpenTuneWeaver/ui
exec python app.py