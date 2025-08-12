#!/bin/bash

# Warte auf Ollama
echo "⏳ Waiting for Ollama service..."
while ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done

echo "✅ Ollama service is ready!"

# Funktion zum sicheren Model-Pull
pull_model() {
    local model=$1
    echo "🔍 Checking model: $model"
    
    if ! ollama list | grep -q "$model"; then
        echo "📥 Pulling $model..."
        if ollama pull "$model"; then
            echo "✅ $model successfully pulled!"
        else
            echo "❌ Failed to pull $model"
            return 1
        fi
    else
        echo "✅ $model already available!"
    fi
}

# Pull Hauptmodell
pull_model "gemma3:12b"

# Optional: Alternative Modelle für Fallback
# pull_model "llama3.2:3b"
# pull_model "mistral:7b"

echo "📋 Available models:"
ollama list

echo "✅ Model pulling complete!"