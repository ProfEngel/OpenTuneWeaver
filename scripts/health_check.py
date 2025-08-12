#!/usr/bin/env python3
"""
Health Check für OpenTuneWeaver mit Ollama
"""

import sys
import requests
import torch
import time

def check_ollama():
    """Prüft ob Ollama läuft und Modell verfügbar ist"""
    try:
        # Check Ollama API
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "Ollama API not responding"
        
        # Check for specific model
        models = response.json().get('models', [])
        model_names = [m.get('name', '') for m in models]
        
        if 'gemma3:12b' not in ' '.join(model_names):
            return False, "Model gemma3:12b not found"
        
        # Test generation
        test_payload = {
            "model": "gemma3:12b",
            "prompt": "Hello",
            "stream": False,
            "options": {"num_predict": 5}
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return True, "Ollama OK with gemma3:12b"
        else:
            return False, f"Generation failed: {response.status_code}"
            
    except Exception as e:
        return False, f"Ollama check failed: {str(e)}"

def check_gradio():
    """Prüft ob Gradio UI läuft"""
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            return True, "Gradio UI OK"
        else:
            return False, f"Gradio status: {response.status_code}"
    except Exception as e:
        return False, f"Gradio check failed: {str(e)}"

def check_cuda():
    """Prüft CUDA Verfügbarkeit"""
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return True, f"CUDA OK: {device_name} ({memory_gb:.1f}GB)"
        else:
            return False, "CUDA not available"
    except Exception as e:
        return False, f"CUDA check failed: {str(e)}"

def main():
    print("🏥 OpenTuneWeaver Health Check")
    print("=" * 50)
    
    checks = [
        ("CUDA", check_cuda),
        ("Ollama", check_ollama),
        ("Gradio UI", check_gradio)
    ]
    
    all_ok = True
    for name, check_func in checks:
        status, message = check_func()
        icon = "✅" if status else "❌"
        print(f"{icon} {name}: {message}")
        all_ok = all_ok and status
    
    print("=" * 50)
    if all_ok:
        print("✅ All systems operational!")
        sys.exit(0)
    else:
        print("❌ Some checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()