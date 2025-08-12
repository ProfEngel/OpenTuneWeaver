#!/bin/bash

# RunPod Setup Helper Script

echo "🚀 OpenTuneWeaver RunPod Setup"
echo "=============================="

# Check if running on RunPod
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "✅ Running on RunPod (Pod ID: $RUNPOD_POD_ID)"
else
    echo "⚠️  Not running on RunPod - Local mode"
fi

# System Info
echo ""
echo "📊 System Information:"
echo "----------------------"
echo "Hostname: $(hostname)"
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $2}')"

# GPU Info
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🎮 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "❌ No GPU detected"
fi

# Network Info
echo ""
echo "🌐 Network Information:"
echo "Public IP: $(curl -s ifconfig.me)"
echo "Ports: 8080 (UI), 11434 (Ollama)"

echo ""
echo "✅ Setup complete! Access UI at:"
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "https://${RUNPOD_POD_ID}-8080.proxy.runpod.net"
else
    echo "http://localhost:8080"
fi