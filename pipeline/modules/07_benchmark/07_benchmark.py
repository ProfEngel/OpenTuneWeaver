#!/usr/bin/env python3

"""
OTW Benchmark Script - VERSION 5.1 + BATCH MODUS + VERBESSERTE ADAPTER-ERKENNUNG

Mit separater Generation und Bewertung zur Vermeidung von VRAM-Konflikten zwischen
Generierungs- und Bewertermodell (Ollama)

HAUPTVERBESSERUNGEN:
- Verbesserte Adapter-Pfad-Erkennung (ohne _merged)
- Nutzt Config-Type aus pipeline_config.json
- Batch-Modus: Alle Antworten generieren, dann alle bewerten
- Kein VRAM-Konflikt zwischen Generierungs- und Bewertermodell
- Aggressives Memory Management zwischen Phasen
- Ollama-spezifisches Memory Cleanup
- Verbesserte Fehlerbehandlung
- Fallback zu Base-Modell + LoRA-Adapter
- Zentrale Konfiguration über PipelineConfigLoader
"""

import json
import os
import time
import statistics
import re
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path

# ========================================
# ZENTRALE KONFIGURATION LADEN
# ========================================

sys.path.append(str(Path(__file__).parent.parent.parent))  # Zum Hauptverzeichnis
from config_loader import PipelineConfigLoader

# Lade Konfiguration
config_loader = PipelineConfigLoader()
bm_config = config_loader.get_benchmark_config()
tokens = config_loader.get_tokens()

# Zeige geladene Konfiguration
print("=" * 60)
print("📋 KONFIGURATION GELADEN (07_benchmark + BATCH MODUS)")
print("=" * 60)
config_loader.print_config_summary()

print(f"\n 📊 Modus: {bm_config.get('mode', 'Unknown')}")
print(f" ⚖️ Evaluator: {bm_config.get('evaluator', {}).get('type', 'Unknown')}")
print(f" 🔑 HF-Token: {'✅' if tokens.get('hf_token') else '❌'}")
print(f" 🚀 Batch-Modus: Aktiviert (vermeidet VRAM-Konflikte)")
print("=" * 60)

# =============================================================================
# CACHE & TOKEN SETUP
# =============================================================================

def setup_hf_cache_optimization():
    """Simple cache optimization to prevent duplicate downloads"""
    cache_base = os.path.expanduser("~/.cache/huggingface")
    os.environ["HF_HOME"] = cache_base
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_base, "hub")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"🗄️ HF Cache optimiert: {cache_base}")
    return cache_base

# Setup cache und setze Tokens
cache_base = setup_hf_cache_optimization()

# Setze HF-Tokens aus zentraler Config
if tokens.get('hf_token'):
    os.environ["HF_TOKEN"] = tokens['hf_token']
    os.environ["HUGGINGFACE_TOKEN"] = tokens['hf_token']
    os.environ["HUGGINGFACE_HUB_TOKEN"] = tokens['hf_token']
    hf_token = tokens['hf_token']
    print(f"🔑 HF_TOKEN aus pipeline_config.json geladen")
else:
    hf_token = None
    print("⚠️ Kein HF_TOKEN in pipeline_config.json gefunden")

# 🚨 KRITISCHER FIX: Deaktiviere PyTorch Dynamo/torch.compile für Stabilität
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch

# Globale Dynamo-Deaktivierung
try:
    import torch._dynamo as dynamo
    dynamo.reset()
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
    print("✅ PyTorch Dynamo/Compile deaktiviert für Stabilität")
except:
    pass

# WICHTIG: Unsloth vor Transformers importieren
try:
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("⚠️ Unsloth nicht verfügbar - verwende nur Transformers")
    UNSLOTH_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import OpenAI

# Matplotlib für Visualisierungen
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from math import pi
    MATPLOTLIB_AVAILABLE = True
    import matplotlib
    matplotlib.use('Agg')
    print("✅ Matplotlib verfügbar - Visualisierungen aktiviert")
except ImportError:
    print("⚠️ Matplotlib nicht verfügbar")
    MATPLOTLIB_AVAILABLE = False

# =============================================================================
# FALLBACK-FUNKTIONEN FÜR BASE + ADAPTER
# =============================================================================

def find_adapter_for_merged(merged_path):
    """Findet den entsprechenden Adapter-Pfad für ein merged Modell"""
    print(f"   🔍 Suche Adapter für: {merged_path}")
    
    # Wenn der Pfad bereits OHNE _merged ist, ist es wahrscheinlich der Adapter
    if not merged_path.endswith('_merged'):
        if os.path.exists(merged_path) and os.path.exists(os.path.join(merged_path, 'adapter_config.json')):
            print(f"   ✅ Pfad ist bereits der Adapter: {merged_path}")
            return merged_path
    
    # Entferne _merged vom Namen
    adapter_path = merged_path.replace('_merged', '')
    
    if os.path.exists(adapter_path):
        # Prüfe ob es ein PEFT-Adapter ist
        if os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
            print(f"   ✅ Adapter gefunden: {adapter_path}")
            return adapter_path
    
    # Wenn merged_path ein absoluter Pfad ist, versuche im gleichen Verzeichnis
    if os.path.isabs(merged_path):
        parent = os.path.dirname(merged_path)
        adapter_name = os.path.basename(merged_path).replace('_merged', '')
        possible_adapter = os.path.join(parent, adapter_name)
        
        if os.path.exists(possible_adapter) and os.path.exists(os.path.join(possible_adapter, 'adapter_config.json')):
            print(f"   ✅ Adapter gefunden: {possible_adapter}")
            return possible_adapter
    
    # Weitere Suchpfade
    parent = os.path.dirname(merged_path) if merged_path else "."
    adapter_name = os.path.basename(merged_path).replace('_merged', '') if merged_path else ""
    
    possible_paths = [
        adapter_path,
        os.path.join(parent, adapter_name),
        os.path.join(parent, f"{adapter_name}_adapter"),
        os.path.join(parent, f"checkpoint-{adapter_name}"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'adapter_config.json')):
            print(f"   ✅ Adapter gefunden: {path}")
            return path
            
    print(f"   ❌ Kein Adapter gefunden für: {merged_path}")
    return None

def get_base_model_for_adapter(adapter_path):
    """Ermittelt das Base-Modell aus der adapter_config.json"""
    config_path = os.path.join(adapter_path, 'adapter_config.json')
    
    if not os.path.exists(config_path):
        return "unsloth/gemma-3n-E2B-it"  # Fallback
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        base_model = config.get('base_model_name_or_path', 'unsloth/gemma-3n-E2B-it')
        print(f"   📋 Base-Modell aus Adapter-Config: {base_model}")
        return base_model
        
    except Exception as e:
        print(f"   ⚠️ Fehler beim Lesen der Adapter-Config: {e}")
        return "unsloth/gemma-3n-E2B-it"  # Fallback

# =============================================================================
# VERBESSERTE MEMORY-FUNKTIONEN MIT OLLAMA-SUPPORT
# =============================================================================

def cleanup_memory():
    """Basic Memory Cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def aggressive_cleanup():
    """Aggressives Memory Cleanup mit mehreren Durchgängen"""
    print("🧹 Aggressives Memory Cleanup...")
    
    # Python Garbage Collection - mehrere Durchgänge
    for i in range(5):
        gc.collect()
    
    # PyTorch CUDA Cleanup
    if torch.cuda.is_available():
        # Cache leeren
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # IPC Memory sammeln
        torch.cuda.ipc_collect()
        
        # Stats zurücksetzen
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Nochmals Cache leeren nach Reset
        torch.cuda.empty_cache()
        
        # Zusätzliche Memory-Bereinigung
        try:
            torch.cuda.reset_cached_memory_stats()
            torch.cuda.synchronize()
        except:
            pass
    
    # Warte kurz, damit OS aufräumen kann
    time.sleep(2)

def cleanup_ollama_memory():
    """Spezifisches Cleanup für Ollama-Server"""
    print("🦙 Ollama Memory-Cleanup...")
    
    # Optional: Ollama API-Call zum Memory-Cleanup
    try:
        import requests
        
        # Verwende die API-URL aus der Konfiguration
        evaluator_config = bm_config.get('evaluator', {})
        base_url = evaluator_config.get('api_base_url', "http://localhost:11434")
        model = evaluator_config.get('model', "gemma3:12b-it-qat")
        
        # Entferne /v1 suffix für direkte Ollama API
        ollama_url = base_url.replace('/v1', '')
        
        requests.post(f"{ollama_url}/api/generate", 
                     json={"model": model, "keep_alive": 0},
                     timeout=5)
        print("   ✅ Ollama Memory-Release angefordert")
    except Exception as e:
        print(f"   ⚠️ Ollama Memory-Release fehlgeschlagen: {e}")
    
    time.sleep(10)  # Warte auf Ollama-Cleanup

def wait_for_vram_free(min_gb=15):
    """Wartet bis genug VRAM frei ist"""
    if not torch.cuda.is_available():
        return True
    
    for attempt in range(30):  # Max 5 Minuten warten
        free_vram = (torch.cuda.get_device_properties(0).total_memory - 
                    torch.cuda.memory_allocated()) / 1024**3
        
        if free_vram >= min_gb:
            print(f"   ✅ {free_vram:.1f} GB VRAM frei")
            return True
        
        print(f"   ⏳ Nur {free_vram:.1f} GB frei, warte... ({attempt+1}/30)")
        time.sleep(10)
    
    print(f"   ❌ Timeout: Nur {free_vram:.1f} GB nach 5 Minuten frei")
    return False

def force_model_unload(model_wrapper):
    """Forciert das komplette Entladen eines Modells"""
    if hasattr(model_wrapper, 'model') and model_wrapper.model:
        # Versuche Modell auf CPU zu verschieben bevor Löschen
        try:
            model_wrapper.model.cpu()
            del model_wrapper.model
        except:
            try:
                del model_wrapper.model
            except:
                pass
        model_wrapper.model = None
    
    if hasattr(model_wrapper, 'tokenizer') and model_wrapper.tokenizer:
        try:
            del model_wrapper.tokenizer
        except:
            pass
        model_wrapper.tokenizer = None
    
    # Cleanup nach Entladen
    aggressive_cleanup()

def print_memory_status(prefix=""):
    """GPU Memory Status mit mehr Details"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = total - allocated
        
        print(f"{prefix}💾 VRAM Status:")
        print(f"   Total: {total:.1f} GB")
        print(f"   Belegt: {allocated:.1f} GB ({allocated/total*100:.1f}%)")
        print(f"   Frei: {free:.1f} GB")
        print(f"   Reserviert: {reserved:.1f} GB")

def check_device_placement(model):
    """Prüft wo das Modell tatsächlich liegt"""
    if hasattr(model, 'hf_device_map'):
        print("📍 Device Map:", model.hf_device_map)
    
    # Prüfe Module-Platzierung
    cpu_modules = []
    gpu_modules = []
    
    for name, param in model.named_parameters():
        if param.device.type == 'cpu':
            cpu_modules.append(name.split('.')[0])
        else:
            gpu_modules.append(name.split('.')[0])
    
    cpu_modules = list(set(cpu_modules))
    gpu_modules = list(set(gpu_modules))
    
    if cpu_modules:
        print(f"⚠️ {len(cpu_modules)} Module auf CPU!")
        print(f"✅ {len(gpu_modules)} Module auf GPU")
        return False
    else:
        print(f"✅ Alle {len(gpu_modules)} Module auf GPU")
        return True

def ultra_cleanup(phase):
    """Ultra-aggressives Cleanup zwischen Phasen"""
    print(f"\n🧹 ULTRA-CLEANUP nach {phase}")
    
    # Mehrfache Cleanup-Durchgänge
    for i in range(5):
        aggressive_cleanup()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"   Cleanup {i+1}/5: {allocated:.1f} GB belegt")
            if allocated < 0.5:
                print("   ✅ Memory erfolgreich bereinigt")
                break
        time.sleep(2)
    
    # Ollama-spezifisches Cleanup
    cleanup_ollama_memory()
    
    print_memory_status("   Nach Ultra-Cleanup: ")

# =============================================================================
# KONFIGURATION AUS ZENTRALER CONFIG - VERBESSERT
# =============================================================================

# Pre-Finetuning (Original)
pre_model_config = bm_config.get('pre_model', {})
PRE_FINETUNING_CONFIG = {
    'model_name': pre_model_config.get('name', "unsloth/gemma-3n-E2B-it"),
    'model_type': pre_model_config.get('type', "transformers"),
    'load_in_4bit': pre_model_config.get('load_in_4bit', False),
    'max_seq_length': pre_model_config.get('max_seq_length', 2048),
    'description': "Pre-Finetuning Modell",
    'chat_template': "gemma-3"
}

# Post-Finetuning (Trainiert) - VERBESSERTE ERKENNUNG
post_model_config = bm_config.get('post_model', {})
ft_config = config_loader.get_finetuning_config()

# Helper function to find the correct path
def find_model_path(path):
    """Findet den korrekten Pfad relativ zum aktuellen Verzeichnis"""
    # Mögliche Pfad-Varianten
    possible_paths = [
        path,  # Original
        f"../{path}",  # Eine Ebene höher
        f"../06_finetuning/{path}",  # Direkt im finetuning Modul
        f"../../{path}",  # Zwei Ebenen höher
        os.path.join("..", "..", path),  # Zwei Ebenen höher (OS-unabhängig)
    ]
    
    # Wenn der Pfad "modules/" enthält, versuche auch ohne
    if "modules/" in path:
        base_path = path.replace("modules/06_finetuning/", "")
        possible_paths.extend([
            f"../06_finetuning/{base_path}",
            f"../../modules/06_finetuning/{base_path}",
        ])
    
    for p in possible_paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

# Verwende den konfigurierten Pfad direkt
post_model_name = post_model_config.get('name', 'auto-detect')
post_model_type = post_model_config.get('type', 'unknown')

# Wenn type "adapter" ist, behandle es entsprechend
if post_model_type == 'adapter':
    print(f"✅ Post-Model als Adapter erkannt: {post_model_name}")
    # Stelle sicher, dass der Pfad OHNE _merged ist
    if post_model_name.endswith('_merged'):
        post_model_name = post_model_name.replace('_merged', '')
        print(f"   📂 Korrigiert zu: {post_model_name}")
    
    # Finde den tatsächlichen Pfad
    actual_path = find_model_path(post_model_name)
    if actual_path:
        post_model_name = actual_path
        print(f"   📁 Tatsächlicher Pfad: {actual_path}")

elif post_model_name == 'auto-detect' or not os.path.exists(post_model_name):
    # Versuche das trainierte Modell zu finden
    custom_model_dir = ft_config.get('custom_model_dir', 'CustomModel')
    model_name = ft_config.get('model_name', 'model')
    
    print(f"🔍 Suche Modell: {model_name} in {custom_model_dir}")
    
    # Prüfe verschiedene Pfad-Kombinationen
    paths_to_check = [
        (f"../06_finetuning/{custom_model_dir}/{model_name}", "adapter"),
        (f"../06_finetuning/{custom_model_dir}/{model_name}_merged", "merged"),
        (f"../../modules/06_finetuning/{custom_model_dir}/{model_name}", "adapter"),
        (f"../../modules/06_finetuning/{custom_model_dir}/{model_name}_merged", "merged"),
        (f"{custom_model_dir}/{model_name}", "adapter"),
        (f"{custom_model_dir}/{model_name}_merged", "merged"),
    ]
    
    # Prüfe welcher Pfad existiert
    for path, model_type in paths_to_check:
        if os.path.exists(path):
            # Prüfe ob es wirklich ein Adapter ist
            if model_type == "adapter" and os.path.exists(os.path.join(path, 'adapter_config.json')):
                post_model_name = os.path.abspath(path)
                post_model_type = 'adapter'
                print(f"✅ Auto-detected Adapter: {post_model_name}")
                break
            # Oder ein merged Modell
            elif model_type == "merged" and (
                os.path.exists(os.path.join(path, 'config.json')) or
                os.path.exists(os.path.join(path, 'pytorch_model.bin')) or
                os.path.exists(os.path.join(path, 'model.safetensors'))
            ):
                post_model_name = os.path.abspath(path)
                post_model_type = 'merged'
                print(f"✅ Auto-detected Merged Model: {post_model_name}")
                break
    else:
        # Kein Modell gefunden - versuche Fallback auf Adapter
        for path, model_type in paths_to_check:
            if model_type == "adapter":
                actual_path = find_model_path(path)
                if actual_path:
                    post_model_name = actual_path
                    post_model_type = 'adapter'
                    print(f"⚠️ Fallback auf Adapter: {post_model_name}")
                    break
        else:
            print(f"❌ Kein Modell gefunden in üblichen Pfaden")
            post_model_name = f"../06_finetuning/{custom_model_dir}/{model_name}"
            post_model_type = 'unknown'
else:
    # Pfad existiert bereits
    actual_path = find_model_path(post_model_name)
    if actual_path:
        post_model_name = actual_path
        print(f"✅ Verwende konfigurierten Pfad: {actual_path}")

POST_FINETUNING_CONFIG = {
    'model_name': post_model_name,
    'model_type': post_model_type,  # "merged", "adapter" oder "unknown"
    'load_in_4bit': post_model_config.get('load_in_4bit', False),
    'max_seq_length': post_model_config.get('max_seq_length', 2048),
    'description': f"Post-Finetuning: {ft_config.get('model_name', 'Unknown')}",
    'chat_template': "gemma-3",
    'base_model': post_model_config.get('base_model', ft_config.get('base_model'))  # Für Adapter
}

# Bewertungsmodell
evaluator_config = bm_config.get('evaluator', {})
EVAL_API_CONFIG = {
    'base_url': evaluator_config.get('api_base_url', "http://localhost:11434/v1"),
    'api_key': evaluator_config.get('api_key', "ollama"),
    'model': evaluator_config.get('model', "gemma3:12b-it-qat"),
    'description': f"Evaluator: {evaluator_config.get('model', 'Unknown')}"
}

# Inference Settings
INFERENCE_CONFIG = {
    'max_new_tokens': bm_config.get('max_new_tokens', 256),
    'temperature': bm_config.get('temperature', 0.3),
    'top_p': bm_config.get('top_p', 0.9),
    'top_k': bm_config.get('top_k', 50),
    'repetition_penalty': bm_config.get('repetition_penalty', 1.1),
    'do_sample': True
}

# Benchmark Mode
BENCHMARK_MODE = bm_config.get('mode', 'comparison')

# =============================================================================
# INTELLIGENTE MODELL-ERKENNUNG MIT FALLBACK-SUPPORT
# =============================================================================

def detect_model_type(model_name: str, config_type: str = None) -> dict:
    """
    Erkennt Modell-Typ und empfiehlt optimale Konfiguration mit Fallback-Info
    
    Args:
        model_name: Pfad zum Modell
        config_type: Typ aus der Config ("merged", "adapter", "unknown")
    """
    model_name_lower = model_name.lower()
    
    # Verwende Config-Type wenn vorhanden
    if config_type == 'adapter':
        # Es ist definitiv ein Adapter
        adapter_path = model_name
        if adapter_path.endswith('_merged'):
            adapter_path = adapter_path.replace('_merged', '')
        
        # Prüfe ob der Adapter-Pfad existiert
        if os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
            fallback_available = True
        else:
            fallback_available = False
            print(f"   ⚠️ Adapter-Pfad nicht gefunden: {adapter_path}")
        
        return {
            'type': 'adapter',
            'use_quantization': False,
            'use_unsloth': False,
            'dtype': torch.float16,
            'reason': 'Adapter aus Config erkannt',
            'fallback_available': fallback_available,
            'fallback_adapter': adapter_path if fallback_available else None
        }
    
    # Prüfe ob es ein Adapter-Verzeichnis ist
    if os.path.exists(model_name) and os.path.exists(os.path.join(model_name, 'adapter_config.json')):
        return {
            'type': 'adapter',
            'use_quantization': False,
            'use_unsloth': False,
            'dtype': torch.float16,
            'reason': 'Adapter-Verzeichnis erkannt',
            'fallback_available': True,
            'fallback_adapter': model_name
        }
    
    # Gemma Modelle - spezielle Behandlung für Cast-Fehler
    if 'gemma' in model_name_lower:
        return {
            'type': 'gemma',
            'use_quantization': False,  # KEINE 4-bit für Gemma!
            'use_unsloth': False,
            'dtype': torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            'reason': 'Gemma erkannt - verwende bfloat16 ohne Quantization',
            'fallback_available': False,
            'fallback_adapter': None
        }
    
    # Merged Modelle
    if '_merged' in model_name_lower:
        # Prüfe ob Adapter verfügbar ist für Fallback
        adapter_path = find_adapter_for_merged(model_name)
        fallback_available = adapter_path is not None
        
        return {
            'type': 'merged',
            'use_quantization': False,
            'use_unsloth': False,
            'dtype': torch.float16,
            'reason': 'Merged Modell - keine Quantization nötig',
            'fallback_available': fallback_available,
            'fallback_adapter': adapter_path
        }
    
    # Standard
    return {
        'type': 'standard',
        'use_quantization': True,
        'use_unsloth': UNSLOTH_AVAILABLE,
        'dtype': torch.float16,
        'reason': 'Standard Modell',
        'fallback_available': False,
        'fallback_adapter': None
    }

def check_model_in_cache(model_name):
    """Check if model is already in cache and return cache path"""
    cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    
    # First check if it's a local path that exists
    if os.path.exists(model_name):
        # It's a valid local path
        return True, model_name
    
    if not os.path.exists(cache_dir):
        return False, None
    
    # Handle HuggingFace format: "unsloth/gemma-3n-E2B-it"
    if "/" in model_name and not model_name.startswith("./") and not model_name.startswith("../"):
        # HuggingFace format - cache name uses "models--" prefix
        cache_model_name = "models--" + model_name.replace("/", "--")
    else:
        # It's a path that doesn't exist
        return False, None
    
    # Suche im Cache-Verzeichnis
    cache_entries = []
    for entry in os.listdir(cache_dir):
        if cache_model_name in entry and os.path.isdir(os.path.join(cache_dir, entry)):
            cache_entries.append(entry)
    
    if cache_entries:
        # Verwende den ersten gefundenen Eintrag
        cache_path = os.path.join(cache_dir, cache_entries[0])
        snapshots_dir = os.path.join(cache_path, "snapshots")
        
        if os.path.exists(snapshots_dir):
            snapshot_dirs = sorted(os.listdir(snapshots_dir))  # Sortiert für Konsistenz
            if snapshot_dirs:
                # Verwende den neuesten Snapshot
                latest_snapshot = os.path.join(snapshots_dir, snapshot_dirs[-1])
                # Prüfe ob Model-Dateien vorhanden sind
                model_files = [f for f in os.listdir(latest_snapshot) 
                              if f.endswith(('.safetensors', '.bin', '.pth'))]
                
                if model_files:
                    total_size = sum(os.path.getsize(os.path.join(latest_snapshot, f)) 
                                   for f in model_files) / (1024**3)
                    print(f"✅ Model im Cache gefunden: {model_name}")
                    print(f"   📊 Größe: {total_size:.1f} GB")
                    print(f"   📁 Cache-Pfad: {latest_snapshot}")
                    return True, latest_snapshot
    
    print(f"❌ Model nicht im Cache: {model_name}")
    return False, None

# =============================================================================
# MODEL WRAPPER mit verbessertem Device Management und FALLBACK
# =============================================================================

class TransformersModelWrapper:
    """Wrapper mit intelligenter Konfiguration, verbessertem Device Management und Fallback"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.chat_template = config.get('chat_template')
        # Übergebe den config_type aus der Pipeline-Config
        config_type = config.get('model_type', None)
        self.model_info = detect_model_type(config['model_name'], config_type)
        
        # Wenn es ein Adapter ist, direkt zum Adapter-Loading
        if self.model_info['type'] == 'adapter' and self.model_info.get('fallback_adapter'):
            # Setze den Adapter-Pfad direkt
            self.config['model_name'] = self.model_info['fallback_adapter']
            self.config['is_adapter'] = True
    
    def _load_base_with_adapter(self):
        """Fallback: Lädt Base-Modell und wendet Adapter an"""
        
        # Adapter-Pfad ist bereits korrekt gesetzt oder muss ermittelt werden
        adapter_path = self.config.get('model_name', '')
        
        # Entferne _merged falls vorhanden
        if adapter_path.endswith('_merged'):
            adapter_path = adapter_path.replace('_merged', '')
            print(f"   📂 Korrigiere Pfad zu: {adapter_path}")
        
        # Prüfe ob Adapter existiert
        if not os.path.exists(adapter_path):
            print(f"   ❌ Adapter-Pfad existiert nicht: {adapter_path}")
            return False
            
        if not os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
            print(f"   ❌ Kein gültiger Adapter in: {adapter_path}")
            # Versuche im gleichen Verzeichnis zu suchen
            parent = os.path.dirname(adapter_path)
            base_name = os.path.basename(adapter_path)
            
            # Versuche ohne _merged suffix
            if '_merged' in base_name:
                base_name = base_name.replace('_merged', '')
                adapter_path = os.path.join(parent, base_name)
                
                if not os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
                    print(f"   ❌ Auch kein Adapter in: {adapter_path}")
                    return False
            else:
                return False
            
        print(f"   ✅ Verwende Adapter: {adapter_path}")
            
        # Base-Modell aus Config oder Adapter-Config ermitteln
        base_model_name = self.config.get('base_model')
        if not base_model_name:
            base_model_name = get_base_model_for_adapter(adapter_path)
        
        print(f"   📁 Base-Modell: {base_model_name}")
        print(f"   🔧 Adapter: {adapter_path}")
        
        try:
            # 2. Lade Base-Modell Tokenizer
            tokenizer_kwargs = {
                'trust_remote_code': True
            }
            
            if hf_token:
                tokenizer_kwargs['token'] = hf_token
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                **tokenizer_kwargs
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 3. Lade Base-Modell
            model_kwargs = {
                'torch_dtype': self.model_info['dtype'],
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }
            
            # WICHTIG: Explizite GPU-Device-Map
            if torch.cuda.is_available():
                free_vram = (torch.cuda.get_device_properties(0).total_memory - 
                           torch.cuda.memory_allocated()) / 1024**3
                if free_vram > 12:  # Genug VRAM für ganzes Modell
                    model_kwargs['device_map'] = {'': 0}  # Alles auf GPU 0
                    print(f"   🎯 Forciere GPU-only Loading (VRAM frei: {free_vram:.1f} GB)")
                else:
                    model_kwargs['device_map'] = 'auto'
                    print(f"   ⚠️ Wenig VRAM ({free_vram:.1f} GB) - Auto device_map")
            else:
                model_kwargs['device_map'] = 'auto'
            
            if hf_token:
                model_kwargs['token'] = hf_token
            
            print("   🔧 OHNE Quantization - Base-Modell Loading")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                **model_kwargs
            )
            
            # 4. Lade und wende Adapter an
            print("   🔧 Lade PEFT Adapter...")
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                
                # 🚨 KRITISCHER FIX: Stelle sicher, dass das gesamte Modell auf GPU ist
                if torch.cuda.is_available():
                    print("   🔧 Verschiebe Adapter-Modell komplett auf GPU...")
                    
                    # Methode 1: Direct .cuda() call
                    try:
                        self.model = self.model.cuda()
                        print("   ✅ Modell auf GPU verschoben")
                    except Exception as cuda_error:
                        print(f"   ⚠️ .cuda() Fehler: {cuda_error}")
                        
                        # Methode 2: Manual device placement
                        try:
                            device = torch.device('cuda:0')
                            self.model = self.model.to(device)
                            print("   ✅ Modell mit .to(device) verschoben")
                        except Exception as to_error:
                            print(f"   ⚠️ .to(device) Fehler: {to_error}")
                            return False
                
                # Prüfe finale Device-Placement
                cpu_params = 0
                gpu_params = 0
                for name, param in self.model.named_parameters():
                    if param.device.type == 'cpu':
                        cpu_params += 1
                    else:
                        gpu_params += 1
                
                if cpu_params > 0:
                    print(f"   ⚠️ WARNUNG: {cpu_params} Parameter noch auf CPU, {gpu_params} auf GPU")
                else:
                    print(f"   ✅ Alle {gpu_params} Parameter auf GPU")
                
                print("   ✅ Base + Adapter erfolgreich geladen!")
                return True
                
            except Exception as adapter_error:
                print(f"   ❌ PEFT Adapter Fehler: {adapter_error}")
                return False
                
        except Exception as e:
            print(f"   ❌ Base+Adapter Fehler: {e}")
            return False
    
    def load_model(self):
        """Lädt Modell mit optimaler Strategie, Device-Map Kontrolle und FALLBACK"""
        model_name = os.path.basename(self.config['model_name'])
        print(f"\n🔄 Lade Modell: {model_name}")
        print(f"   📋 {self.model_info['reason']}")
        print(f"   📁 Vollständiger Pfad: {self.config['model_name']}")
        
        # Cleanup vor dem Laden
        cleanup_memory()
        print_memory_status("   Vor Laden: ")
        
        # Wenn es bereits als Adapter erkannt wurde, direkt laden
        if self.config.get('is_adapter') or self.model_info['type'] == 'adapter':
            print("   🔧 Lade als Adapter...")
            return self._load_base_with_adapter()
        
        # Prüfe ob der Pfad existiert
        model_path = self.config['model_name']
        
        if os.path.exists(model_path):
            # Lokaler Pfad existiert
            print(f"   ✅ Lokaler Pfad existiert: {model_path}")
            use_local = True
        else:
            # Prüfe ob Modell im Cache ist (nur für HuggingFace Modelle)
            is_cached, cache_path = check_model_in_cache(model_path)
            if is_cached and cache_path:
                model_path = cache_path
                use_local = True
                print(f"   🗄️ Verwende Cache: {cache_path}")
            else:
                print(f"   ❌ Modell nicht gefunden: {model_path}")
                
                # Versuche Fallback auf Adapter
                if self.model_info.get('fallback_available', False):
                    print("   🔄 Fallback: Versuche Base-Modell + Adapter...")
                    return self._load_base_with_adapter()
                else:
                    raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            # Versuche merged Modell zu laden
            return self._load_merged_model(model_path, use_local)
            
        except Exception as merged_error:
            error_msg = str(merged_error)
            print(f"   ⚠️ Modell-Lade Fehler: {error_msg[:200]}...")
            
            # Prüfe Fallback-Szenarien
            if self.model_info.get('fallback_available', False):
                print("   🔄 Fallback: Versuche Base-Modell + Adapter...")
                return self._load_base_with_adapter()
            
            elif "CUDA out of memory" in error_msg:
                print("   🚨 CUDA OOM - versuche mit Quantization...")
                return self._try_quantization_fallback(model_path, use_local)
            
            else:
                print("   ❌ Kein geeigneter Fallback verfügbar")
                raise merged_error
    
    def _try_quantization_fallback(self, model_path, use_local):
        """Fallback mit Quantization bei Memory-Problemen"""
        print("   🔧 Quantization-Fallback aktiviert")
        
        # Emergency Memory Recovery
        aggressive_cleanup()
        
        # Erzwinge Quantization
        original_quantization = self.model_info['use_quantization']
        original_4bit = self.config.get('load_in_4bit', False)
        
        self.model_info['use_quantization'] = True
        self.config['load_in_4bit'] = True
        
        try:
            result = self._load_merged_model(model_path, use_local)
            print("   ✅ Quantization-Fallback erfolgreich!")
            return result
        except Exception as e:
            print(f"   ❌ Auch Quantization-Fallback gescheitert: {e}")
            # Restore original settings
            self.model_info['use_quantization'] = original_quantization
            self.config['load_in_4bit'] = original_4bit
            return False
    
    def _load_merged_model(self, model_path, use_local):
        """Lädt das merged Modell (ursprüngliche Logik mit Verbesserungen)"""
        
        # Wichtig: Prüfe ob der Pfad wirklich existiert
        if not os.path.exists(model_path):
            print(f"   ❌ Pfad existiert nicht: {model_path}")
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        print(f"   📂 Lade von lokalem Pfad: {model_path}")
        
        # Tokenizer laden
        tokenizer_kwargs = {
            'trust_remote_code': True
        }
        
        # Füge Token hinzu wenn verfügbar
        if hf_token:
            tokenizer_kwargs['token'] = hf_token
        
        # Bei lokalem Pfad immer von dort laden
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                **tokenizer_kwargs
            )
        except Exception as e:
            print(f"   ⚠️ Tokenizer-Fehler: {e}")
            # Fallback: Versuche Base-Model Tokenizer
            if self.config.get('base_model'):
                print(f"   🔄 Lade Base-Model Tokenizer: {self.config['base_model']}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config['base_model'],
                    **tokenizer_kwargs
                )
            else:
                raise e
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Modell-Konfiguration mit expliziter Device-Map
        model_kwargs = {
            'torch_dtype': self.model_info['dtype'],
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
        }
        
        # WICHTIG: Explizite GPU-Device-Map wenn genug VRAM
        if torch.cuda.is_available():
            free_vram = (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_allocated()) / 1024**3
            if free_vram > 12:  # Genug VRAM für ganzes Modell
                model_kwargs['device_map'] = {'': 0}  # Alles auf GPU 0
                print(f"   🎯 Forciere GPU-only Loading (VRAM frei: {free_vram:.1f} GB)")
            else:
                print(f"   ⚠️ Wenig VRAM ({free_vram:.1f} GB) - Auto device_map")
                model_kwargs['device_map'] = 'auto'
        else:
            model_kwargs['device_map'] = 'auto'
        
        # Füge Token hinzu wenn verfügbar
        if hf_token:
            model_kwargs['token'] = hf_token
        
        # Quantization nur wenn empfohlen
        if self.model_info['use_quantization'] and self.config.get('load_in_4bit'):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.model_info['dtype'],
                bnb_4bit_use_double_quant=True
            )
            model_kwargs['quantization_config'] = bnb_config
            print("   🔧 Mit 4-bit Quantization")
        else:
            print(f"   🔧 OHNE Quantization - {self.model_info['dtype']}")
        
        # Modell laden - direkt vom lokalen Pfad
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
        except Exception as e:
            print(f"   ❌ Fehler beim Laden des Modells: {e}")
            raise e
        
        # Prüfe Device-Placement
        is_fully_on_gpu = check_device_placement(self.model)
        if not is_fully_on_gpu:
            print("   ⚠️ WARNUNG: Modell teilweise auf CPU - Performance wird leiden!")
        
        print(f"   ✅ Modell geladen")
        print_memory_status("   Nach Laden: ")
        
        return True
    
    def generate_answer(self, question: str, question_id: str) -> str:
        """Generiert Antwort mit Device-Fehler Workaround"""
        if not self.model or not self.tokenizer:
            return "Fehler: Modell nicht geladen"
        
        try:
            # Chat Template
            if self.chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": question}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = f"Frage: {question}\nAntwort:"
            
            # Tokenisierung
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config['max_seq_length'] - INFERENCE_CONFIG['max_new_tokens'],
            )
            
            # 🚨 DEVICE-FIX: Stelle sicher, dass inputs auf gleichem Device wie Modell sind
            if torch.cuda.is_available():
                # Ermittle Device des Modells
                try:
                    model_device = next(self.model.parameters()).device
                    # Verschiebe Inputs auf gleiches Device
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                except:
                    # Fallback: Versuche CUDA
                    inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generation mit verbesserter Fehlerbehandlung
            with torch.no_grad():
                try:
                    if 'gemma' in self.config['model_name'].lower():
                        # Spezielle Behandlung für Gemma
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=INFERENCE_CONFIG['max_new_tokens'],
                                temperature=INFERENCE_CONFIG['temperature'],
                                top_p=INFERENCE_CONFIG['top_p'],
                                top_k=INFERENCE_CONFIG['top_k'],
                                repetition_penalty=INFERENCE_CONFIG['repetition_penalty'],
                                do_sample=INFERENCE_CONFIG['do_sample'],
                                pad_token_id=self.tokenizer.eos_token_id,
                            )
                    else:
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=INFERENCE_CONFIG['max_new_tokens'],
                            temperature=INFERENCE_CONFIG['temperature'],
                            top_p=INFERENCE_CONFIG['top_p'],
                            top_k=INFERENCE_CONFIG['top_k'],
                            repetition_penalty=INFERENCE_CONFIG['repetition_penalty'],
                            do_sample=INFERENCE_CONFIG['do_sample'],
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                except RuntimeError as device_error:
                    if "device" in str(device_error).lower():
                        print(f"   ⚠️ Device-Fehler: Versuche Reparatur...")
                        # Versuche alle Modell-Parameter auf GPU zu verschieben
                        if torch.cuda.is_available():
                            try:
                                self.model = self.model.cuda()
                                inputs = {k: v.cuda() for k, v in inputs.items()}
                            except:
                                pass
                        
                        # Vereinfachter zweiter Versuch
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=INFERENCE_CONFIG['max_new_tokens'],
                            temperature=0.7,
                            do_sample=False,  # Vereinfachte Generation
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    else:
                        raise device_error
            
            # Dekodierung
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return answer.strip()
            
        except Exception as e:
            print(f"   ⚠️ Generation-Fehler: {str(e)[:100]}")
            return "Antwort konnte nicht generiert werden."
    
    def unload_model(self):
        """Entlädt Modell vollständig"""
        print("   🗑️ Entlade Modell...")
        if self.model:
            # Versuche auf CPU zu verschieben vor dem Löschen
            try:
                self.model.cpu()
            except:
                pass
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Aggressives Cleanup
        aggressive_cleanup()
        print("   ✅ Modell entladen und Memory bereinigt")

# =============================================================================
# BENCHMARK KLASSE mit BATCH-MODUS
# =============================================================================

class TransformersBenchmark:
    def __init__(self, answer_config: dict, eval_config: dict, use_api_eval: bool = True):
        self.answer_config = answer_config
        self.eval_config = eval_config
        self.use_api_eval = use_api_eval
        self.answer_model = None
        self.eval_client = None
        
        if self.use_api_eval:
            self.eval_client = OpenAI(
                api_key=eval_config['api_key'],
                base_url=eval_config['base_url']
            )
        
        self.questions_data = None
        self.results = []
        
        print(f"🔧 Benchmark initialisiert")
        print(f"   📝 Modell: {os.path.basename(answer_config['model_name'])}")
        print(f"   ⚖️  Bewerter: {'API' if use_api_eval else 'Lokal'}")
    
    def load_questions(self, file_path: str = None):
        """Lädt Fragen"""
        if file_path is None:
            file_path = bm_config.get('questions_file', "BENCHMARKFRAGEN/benchmark_fragen_complete.json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.questions_data = json.load(f)
            
            total = sum(len(k['fragen']) for k in self.questions_data['kategorien'])
            print(f"✅ {total} Fragen geladen aus {file_path}")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Laden: {e}")
            return False
    
    def evaluate_answer(self, question: str, generated: str, correct: str, question_id: str):
        """Bewertet Antwort"""
        try:
            prompt = f"""Bewerte diese Antwort auf einer Skala von 0-10.

FRAGE: {question}

MUSTERLÖSUNG: {correct}

GEGEBENE ANTWORT: {generated}

Bewerte nach:
1. Vollständigkeit (0-2)
2. Korrektheit (0-2)
3. Präzision (0-2)
4. Keine Halluzinationen (0-2)
5. Struktur (0-2)

Gib nur die GESAMTPUNKTE als Zahl zwischen 0 und 10 zurück."""
            
            if self.use_api_eval and self.eval_client:
                response = self.eval_client.chat.completions.create(
                    model=self.eval_config['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100
                )
                
                text = response.choices[0].message.content.strip()
                
                # Extrahiere Score
                numbers = re.findall(r'\d+', text)
                if numbers:
                    score = int(numbers[-1])  # Nimm die letzte Zahl
                    score = max(0, min(10, score))
                else:
                    score = 5
                
                return score, text
            else:
                return 5, "Lokale Bewertung nicht implementiert"
                
        except Exception as e:
            print(f"   ⚠️ Bewertungsfehler: {e}")
            return 0, "Fehler"
    
    def _generate_answers_only(self, description):
        """Generiert nur Antworten ohne Bewertung"""
        print(f"\n🚀 Generiere {description} Antworten")
        
        # Lade Modell
        self.answer_model = TransformersModelWrapper(self.answer_config)
        if not self.answer_model.load_model():
            print(f"❌ {description}-Modell konnte nicht geladen werden")
            return []
        
        answers = []
        total_questions = 0
        
        for kategorie_data in self.questions_data['kategorien']:
            kategorie = kategorie_data['kategorie']
            print(f"\n📂 {kategorie}")
            
            for frage_data in kategorie_data['fragen']:
                total_questions += 1
                qid = frage_data['id']
                question = frage_data['frage']
                correct = frage_data['antwort']
                
                print(f"   {total_questions}. {question[:60]}...")
                
                generated = self.answer_model.generate_answer(question, qid)
                
                answers.append({
                    'category': kategorie,
                    'id': qid,
                    'question': question,
                    'correct': correct,
                    'generated': generated
                })
        
        # Modell sofort entladen
        self.answer_model.unload_model()
        print(f"✅ {description}-Modell entladen")
        
        return answers
    
    def _evaluate_answers_batch(self, answers):
        """Bewertet alle Antworten in einem Batch"""
        results = []
        total = len(answers)
        
        for i, answer_data in enumerate(answers, 1):
            print(f"   {i}/{total}", end="\r")
            
            score, reasoning = self.evaluate_answer(
                answer_data['question'],
                answer_data['generated'],
                answer_data['correct'],
                answer_data['id']
            )
            
            results.append({
                'category': answer_data['category'],
                'id': answer_data['id'],
                'question': answer_data['question'],
                'generated': answer_data['generated'],
                'score': score,
                'reasoning': reasoning
            })
        
        return results
    
    def _calculate_score(self, results):
        """Berechnet Prozentscore"""
        scores = [r['score'] for r in results]
        return (sum(scores) / (len(scores) * 10)) * 100 if scores else 0
    
    def _show_results(self, results):
        """Zeigt finale Ergebnisse"""
        if 'pre' in results and 'post' in results:
            pre_score = results['pre']['percentage_score']
            post_score = results['post']['percentage_score']
            improvement = post_score - pre_score
            
            print(f"\n🏆 VOLLSTÄNDIGES ERGEBNIS:")
            print(f"   Pre: {pre_score:.1f}%")
            print(f"   Post: {post_score:.1f}%")
            print(f"   📈 Verbesserung: {improvement:+.1f}%")
            
            if improvement > 0:
                print("   ✅ Finetuning erfolgreich!")
            else:
                print("   ⚠️ Keine Verbesserung messbar")
                
        elif 'post' in results:
            post_score = results['post']['percentage_score']
            print(f"\n⚠️  TEILWEISES ERGEBNIS (nur POST):")
            print(f"   Post: {post_score:.1f}%")
            
        elif 'pre' in results:
            pre_score = results['pre']['percentage_score']
            print(f"\n⚠️  TEILWEISES ERGEBNIS (nur PRE):")
            print(f"   Pre: {pre_score:.1f}%")
    
    def run_benchmark(self, description=""):
        """Führt Einzelbenchmark durch"""
        if not self.questions_data:
            print("❌ Keine Fragen geladen")
            return None
        
        print(f"\n🚀 Starte Benchmark {description}")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Phase 1: Antworten generieren
        print("\n📝 Phase 1: Antworten generieren")
        answers = self._generate_answers_only(description)
        
        if not answers:
            return None
        
        # Phase 2: Bewerten
        print(f"\n⚖️  Phase 2: Bewerten")
        self.results = self._evaluate_answers_batch(answers)
        
        # Statistiken
        duration = datetime.now() - start_time
        percentage = self._calculate_score(self.results)
        scores = [r['score'] for r in self.results]
        
        print(f"\n\n✅ FERTIG")
        print(f"   ⏱️  {duration}")
        print(f"   📊 Score: {percentage:.1f}%")
        print(f"   📈 Durchschnitt: {statistics.mean(scores):.1f}/10")
        
        return {
            'config': self.answer_config,
            'results': self.results,
            'percentage_score': percentage,
            'duration': duration
        }
    
    def run_comparison(self, pre_config, post_config):
        """Vergleichsbenchmark mit BATCH-MODUS zur VRAM-Konflikt-Vermeidung"""
        print("🎯 VERGLEICHSBENCHMARK - BATCH MODUS")
        print("=" * 60)
        
        all_answers = {}  # Sammelt alle Antworten
        results = {}
        
        # ========================================
        # PHASE 1: ALLE ANTWORTEN GENERIEREN
        # ========================================
        print("\n📝 PHASE 1: ANTWORTEN GENERIEREN")
        print("=" * 40)
        print("🎯 Generiere alle Antworten OHNE Ollama-Bewerter")
        print("   (Vermeidet VRAM-Konflikt zwischen Modellen)")
        
        # Pre-Finetuning Antworten
        print("\n🔄 Generiere PRE-Finetuning Antworten...")
        print_memory_status("Vor Pre-Generation: ")
        
        pre_benchmark = TransformersBenchmark(pre_config, self.eval_config, use_api_eval=False)
        if pre_benchmark.load_questions():
            pre_answers = pre_benchmark._generate_answers_only("PRE")
            all_answers['pre'] = pre_answers
            print(f"✅ {len(pre_answers)} PRE-Antworten generiert")
        
        # Aggressives Cleanup zwischen Modellen
        ultra_cleanup("Pre-Generation")
        
        # Post-Finetuning Antworten  
        print("\n🔄 Generiere POST-Finetuning Antworten...")
        print_memory_status("Vor Post-Generation: ")
        
        post_benchmark = TransformersBenchmark(post_config, self.eval_config, use_api_eval=False)
        if post_benchmark.load_questions():
            post_answers = post_benchmark._generate_answers_only("POST")
            all_answers['post'] = post_answers
            print(f"✅ {len(post_answers)} POST-Antworten generiert")
        
        # Finales Cleanup vor Bewertung
        ultra_cleanup("Post-Generation")
        
        # ========================================
        # PHASE 2: ALLE ANTWORTEN BEWERTEN
        # ========================================
        print("\n⚖️  PHASE 2: BATCH-BEWERTUNG")
        print("=" * 40)
        print("🤖 Starte Ollama-Bewerter...")
        print("   (Jetzt ist VRAM frei für Bewertung)")
        
        # Warte bis genug VRAM frei ist
        if not wait_for_vram_free(min_gb=12):
            print("❌ Nicht genug VRAM für Ollama-Bewerter")
            return results
        
        # Jetzt ist VRAM frei für Ollama
        evaluator = TransformersBenchmark(pre_config, self.eval_config, use_api_eval=True)
        
        # Bewerte alle PRE-Antworten
        if 'pre' in all_answers:
            print("\n📊 Bewerte PRE-Antworten...")
            pre_results = evaluator._evaluate_answers_batch(all_answers['pre'])
            results['pre'] = {
                'config': pre_config,
                'results': pre_results,
                'percentage_score': evaluator._calculate_score(pre_results),
                'duration': None
            }
        
        # Bewerte alle POST-Antworten
        if 'post' in all_answers:
            print("\n📊 Bewerte POST-Antworten...")
            post_results = evaluator._evaluate_answers_batch(all_answers['post'])
            results['post'] = {
                'config': post_config,
                'results': post_results,
                'percentage_score': evaluator._calculate_score(post_results),
                'duration': None
            }
        
        # ========================================
        # ERGEBNISSE
        # ========================================
        evaluator._show_results(results)
        
        # Speichern
        if results:
            os.makedirs("OUTPUT", exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if 'pre' in results and 'post' in results:
                filename = f"OUTPUT/comparison_batch_{timestamp}.json"
            elif 'post' in results:
                filename = f"OUTPUT/post_only_batch_{timestamp}.json"
            else:
                filename = f"OUTPUT/pre_only_batch_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 Gespeichert: {filename}")
        
        return results

# =============================================================================
# HAUPTFUNKTION
# =============================================================================

def main():
    """Hauptfunktion mit zentraler Konfiguration und Batch-Modus"""
    print("🎯 OTW Benchmark Tool - MIT ZENTRALER CONFIG + BATCH MODUS")
    print("✅ Konfiguration aus pipeline_config.json")
    print("✅ Batch-Modus gegen VRAM-Konflikte")
    print("✅ Base + Adapter Fallback")
    print("✅ Ollama-kompatibles Memory Management")
    print("=" * 60)
    
    # Sofortiger Startup-Cleanup
    print("\n🧹 Startup Memory Cleanup...")
    aggressive_cleanup()
    
    # Zeige Cache-Status
    print(f"\n📁 Cache-Verzeichnis: {cache_base}")
    print_memory_status("Start-Status: ")
    
    # Verwende Evaluator-Typ aus Config
    use_api = evaluator_config.get('type', 'api') == 'api'
    
    # Benchmark basierend auf Mode aus Config
    if BENCHMARK_MODE == "comparison":
        # Vergleich mit Batch-Modus
        print(f"\n🔄 Starte Batch-Vergleichsbenchmark")
        print(f"   Pre-Model: {PRE_FINETUNING_CONFIG['model_name']}")
        print(f"   Post-Model: {POST_FINETUNING_CONFIG['model_name']}")
        print(f"   🎯 BATCH-MODUS aktiviert:")
        print(f"      1. Generiere PRE-Antworten → Modell entladen")
        print(f"      2. Generiere POST-Antworten → Modell entladen") 
        print(f"      3. Starte Ollama-Bewerter → Bewerte ALLE")
        print(f"      ✅ Kein VRAM-Konflikt zwischen Modellen!")
        
        benchmark = TransformersBenchmark(
            PRE_FINETUNING_CONFIG,
            EVAL_API_CONFIG,
            use_api
        )
        if benchmark.load_questions():
            benchmark.run_comparison(PRE_FINETUNING_CONFIG, POST_FINETUNING_CONFIG)
            
    elif BENCHMARK_MODE == "post_only":
        # Nur Post
        benchmark = TransformersBenchmark(
            POST_FINETUNING_CONFIG,
            EVAL_API_CONFIG,
            use_api
        )
        if benchmark.load_questions():
            print(f"\n📝 Starte Post-Benchmark")
            print(f"   Model: {POST_FINETUNING_CONFIG['model_name']}")
            benchmark.run_benchmark("POST-FINETUNING")
            
    elif BENCHMARK_MODE == "pre_only":
        # Nur Pre
        benchmark = TransformersBenchmark(
            PRE_FINETUNING_CONFIG,
            EVAL_API_CONFIG,
            use_api
        )
        if benchmark.load_questions():
            print(f"\n📝 Starte Pre-Benchmark")
            print(f"   Model: {PRE_FINETUNING_CONFIG['model_name']}")
            benchmark.run_benchmark("PRE-FINETUNING")
            
    else:
        print(f"❌ Unbekannter Benchmark-Modus: {BENCHMARK_MODE}")
        print("   Gültige Modi: comparison, post_only, pre_only")
        return
    
    # Final Cleanup
    aggressive_cleanup()
    print_memory_status("\nEnd-Status: ")
    
    # Re-enable PyTorch Compile
    os.environ.pop('TORCH_COMPILE_DISABLE', None)
    print("\n✅ Fertig - PyTorch Compile wieder aktiviert")

if __name__ == "__main__":
    main()