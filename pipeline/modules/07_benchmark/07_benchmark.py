#!/usr/bin/env python3

"""
OTW Benchmark Script - VERSION 5.2 + AUTO-DOWNLOAD + DYNAMIC BASE MODEL

With automatic download of base model and dynamic detection
from finetuning configuration for consistent benchmarks.

MAIN IMPROVEMENTS:
- Automatic download of base model if not in cache
- Dynamic detection of used base model from finetuning config
- Consistent comparison: same base model for PRE and POST
- Batch mode to avoid VRAM conflicts remains preserved
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
# LOAD CENTRAL CONFIGURATION
# ========================================

sys.path.append(str(Path(__file__).parent.parent.parent))  # To main directory
from config_loader import PipelineConfigLoader

# Load configuration
config_loader = PipelineConfigLoader()
bm_config = config_loader.get_benchmark_config()
tokens = config_loader.get_tokens()

# Show loaded configuration
print("=" * 60)
print("📋 CONFIGURATION LOADED (07_benchmark + BATCH MODE)")
print("=" * 60)
config_loader.print_config_summary()

print(f"\n📊 Mode: {bm_config.get('mode', 'Unknown')}")
print(f"⚖️ Evaluator: {bm_config.get('evaluator', {}).get('type', 'Unknown')}")
print(f"🔑 HF-Token: {'✅' if tokens.get('hf_token') else '❌'}")
print(f"🚀 Batch mode: Activated (avoids VRAM conflicts)")
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
    print(f"🗄️ HF Cache optimized: {cache_base}")
    return cache_base

# Setup cache and set tokens
cache_base = setup_hf_cache_optimization()

# Set HF tokens from central config
if tokens.get('hf_token'):
    os.environ["HF_TOKEN"] = tokens['hf_token']
    os.environ["HUGGINGFACE_TOKEN"] = tokens['hf_token']
    os.environ["HUGGINGFACE_HUB_TOKEN"] = tokens['hf_token']
    hf_token = tokens['hf_token']
    print(f"🔑 HF_TOKEN loaded from pipeline_config.json")
else:
    hf_token = None
    print("⚠️ No HF_TOKEN found in pipeline_config.json")

# 🚨 CRITICAL FIX: Disable PyTorch Dynamo/torch.compile for stability
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch

# Global Dynamo deactivation
try:
    import torch._dynamo as dynamo
    dynamo.reset()
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
    print("✅ PyTorch Dynamo/Compile disabled for stability")
except:
    pass

# IMPORTANT: Import Unsloth before Transformers
try:
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("⚠️ Unsloth not available - using Transformers only")
    UNSLOTH_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import OpenAI

# Matplotlib for visualizations
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from math import pi
    MATPLOTLIB_AVAILABLE = True
    import matplotlib
    matplotlib.use('Agg')
    print("✅ Matplotlib available - visualizations enabled")
except ImportError:
    print("⚠️ Matplotlib not available")
    MATPLOTLIB_AVAILABLE = False

# =============================================================================
# DYNAMIC BASE MODEL DETECTION
# =============================================================================

def get_base_model_from_finetuning():
    """Determines the base model used in finetuning from config"""
    ft_config = config_loader.get_finetuning_config()
    base_model = ft_config.get('base_model', 'unsloth/gemma-3n-E2B-it')
    
    print(f"🔍 Base model from finetuning config: {base_model}")
    return base_model

def find_trained_model():
    """Finds the trained model (adapter or merged)"""
    ft_config = config_loader.get_finetuning_config()
    custom_model_dir = ft_config.get('custom_model_dir', 'CustomModel')
    model_name = ft_config.get('model_name', 'model')
    
    print(f"🔍 Looking for trained model: {model_name} in {custom_model_dir}")
    
    # Check various path combinations
    paths_to_check = [
        (f"../06_finetuning/{custom_model_dir}/{model_name}", "adapter"),
        (f"../06_finetuning/{custom_model_dir}/{model_name}_merged", "merged"),
        (f"../../modules/06_finetuning/{custom_model_dir}/{model_name}", "adapter"),
        (f"../../modules/06_finetuning/{custom_model_dir}/{model_name}_merged", "merged"),
        (f"{custom_model_dir}/{model_name}", "adapter"),
        (f"{custom_model_dir}/{model_name}_merged", "merged"),
    ]
    
    for path, model_type in paths_to_check:
        if os.path.exists(path):
            if model_type == "adapter" and os.path.exists(os.path.join(path, 'adapter_config.json')):
                return os.path.abspath(path), "adapter"
            elif model_type == "merged" and (
                os.path.exists(os.path.join(path, 'config.json')) or
                os.path.exists(os.path.join(path, 'model.safetensors'))
            ):
                return os.path.abspath(path), "merged"
    
    # Fallback to adapter if nothing found
    default_path = f"../06_finetuning/{custom_model_dir}/{model_name}"
    return default_path, "adapter"

# =============================================================================
# FALLBACK FUNCTIONS FOR BASE + ADAPTER
# =============================================================================

def find_adapter_for_merged(merged_path):
    """Finds the corresponding adapter path for a merged model"""
    print(f"   🔍 Looking for adapter for: {merged_path}")
    
    # If path doesn't end with _merged, it's probably already the adapter
    if not merged_path.endswith('_merged'):
        if os.path.exists(merged_path) and os.path.exists(os.path.join(merged_path, 'adapter_config.json')):
            print(f"   ✅ Path is already the adapter: {merged_path}")
            return merged_path
    
    # Remove _merged from name
    adapter_path = merged_path.replace('_merged', '')
    
    if os.path.exists(adapter_path):
        # Check if it's a PEFT adapter
        if os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
            print(f"   ✅ Adapter found: {adapter_path}")
            return adapter_path
    
    # If merged_path is absolute, try in same directory
    if os.path.isabs(merged_path):
        parent = os.path.dirname(merged_path)
        adapter_name = os.path.basename(merged_path).replace('_merged', '')
        possible_adapter = os.path.join(parent, adapter_name)
        
        if os.path.exists(possible_adapter) and os.path.exists(os.path.join(possible_adapter, 'adapter_config.json')):
            print(f"   ✅ Adapter found: {possible_adapter}")
            return possible_adapter
    
    # Additional search paths
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
            print(f"   ✅ Adapter found: {path}")
            return path
            
    print(f"   ❌ No adapter found for: {merged_path}")
    return None

def get_base_model_for_adapter(adapter_path):
    """Determines base model from adapter_config.json"""
    config_path = os.path.join(adapter_path, 'adapter_config.json')
    
    if not os.path.exists(config_path):
        # Fallback to base model from finetuning config
        return get_base_model_from_finetuning()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        base_model = config.get('base_model_name_or_path')
        if not base_model:
            # Fallback to finetuning config
            base_model = get_base_model_from_finetuning()
        
        print(f"   📋 Base model from adapter config: {base_model}")
        return base_model
        
    except Exception as e:
        print(f"   ⚠️ Error reading adapter config: {e}")
        return get_base_model_from_finetuning()

# =============================================================================
# IMPROVED MEMORY FUNCTIONS WITH OLLAMA SUPPORT
# =============================================================================

def cleanup_memory():
    """Basic Memory Cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def aggressive_cleanup():
    """Aggressive memory cleanup with multiple passes"""
    print("🧹 Aggressive memory cleanup...")
    
    # Python Garbage Collection - multiple passes
    for i in range(5):
        gc.collect()
    
    # PyTorch CUDA Cleanup
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Collect IPC memory
        torch.cuda.ipc_collect()
        
        # Reset stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Clear cache again after reset
        torch.cuda.empty_cache()
        
        # Additional memory cleanup
        try:
            torch.cuda.reset_cached_memory_stats()
            torch.cuda.synchronize()
        except:
            pass
    
    # Wait briefly for OS to clean up
    time.sleep(2)

def cleanup_ollama_memory():
    """Specific cleanup for Ollama server"""
    print("🦙 Ollama memory cleanup...")
    
    # Optional: Ollama API call for memory cleanup
    try:
        import requests
        
        # Use API URL from configuration
        evaluator_config = bm_config.get('evaluator', {})
        base_url = evaluator_config.get('api_base_url', "http://localhost:11434")
        model = evaluator_config.get('model', "gpt-oss:20b")
        
        # Remove /v1 suffix for direct Ollama API
        ollama_url = base_url.replace('/v1', '')
        
        requests.post(f"{ollama_url}/api/generate", 
                     json={"model": model, "keep_alive": 0},
                     timeout=5)
        print("   ✅ Ollama memory release requested")
    except Exception as e:
        print(f"   ⚠️ Ollama memory release failed: {e}")
    
    time.sleep(10)  # Wait for Ollama cleanup

def wait_for_vram_free(min_gb=15):
    """Waits until enough VRAM is free"""
    if not torch.cuda.is_available():
        return True
    
    for attempt in range(30):  # Max 5 minutes wait
        free_vram = (torch.cuda.get_device_properties(0).total_memory - 
                    torch.cuda.memory_allocated()) / 1024**3
        
        if free_vram >= min_gb:
            print(f"   ✅ {free_vram:.1f} GB VRAM free")
            return True
        
        print(f"   ⏳ Only {free_vram:.1f} GB free, waiting... ({attempt+1}/30)")
        time.sleep(10)
    
    print(f"   ❌ Timeout: Only {free_vram:.1f} GB after 5 minutes free")
    return False

def force_model_unload(model_wrapper):
    """Forces complete unloading of a model"""
    if hasattr(model_wrapper, 'model') and model_wrapper.model:
        # Try to move model to CPU before deleting
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
    
    # Cleanup after unloading
    aggressive_cleanup()

def print_memory_status(prefix=""):
    """GPU Memory Status with more details"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = total - allocated
        
        print(f"{prefix}💾 VRAM Status:")
        print(f"   Total: {total:.1f} GB")
        print(f"   Used: {allocated:.1f} GB ({allocated/total*100:.1f}%)")
        print(f"   Free: {free:.1f} GB")
        print(f"   Reserved: {reserved:.1f} GB")

def check_device_placement(model):
    """Checks where the model actually resides"""
    if hasattr(model, 'hf_device_map'):
        print("🔍 Device Map:", model.hf_device_map)
    
    # Check module placement
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
        print(f"⚠️ {len(cpu_modules)} modules on CPU!")
        print(f"✅ {len(gpu_modules)} modules on GPU")
        return False
    else:
        print(f"✅ All {len(gpu_modules)} modules on GPU")
        return True

def ultra_cleanup(phase):
    """Ultra-aggressive cleanup between phases"""
    print(f"\n🧹 ULTRA-CLEANUP after {phase}")
    
    # Multiple cleanup passes
    for i in range(5):
        aggressive_cleanup()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"   Cleanup {i+1}/5: {allocated:.1f} GB used")
            if allocated < 0.5:
                print("   ✅ Memory successfully cleared")
                break
        time.sleep(2)
    
    # Ollama-specific cleanup
    cleanup_ollama_memory()
    
    print_memory_status("   After ultra-cleanup: ")

# =============================================================================
# CONFIGURATION FROM CENTRAL CONFIG - DYNAMICALLY IMPROVED
# =============================================================================

# Determine used base model from finetuning config
BASE_MODEL_NAME = get_base_model_from_finetuning()

# Pre-Finetuning (Base model from finetuning)
PRE_FINETUNING_CONFIG = {
    'model_name': BASE_MODEL_NAME,  # Dynamic from finetuning config
    'model_type': "base",
    'load_in_4bit': False,  # Gemma without quantization
    'max_seq_length': 2048,
    'description': f"Pre-Finetuning: {BASE_MODEL_NAME}",
    'chat_template': "gemma-3"
}

# Post-Finetuning (Trained model)
trained_model_path, trained_model_type = find_trained_model()
POST_FINETUNING_CONFIG = {
    'model_name': trained_model_path,
    'model_type': trained_model_type,
    'load_in_4bit': False,
    'max_seq_length': 2048,
    'description': f"Post-Finetuning: Trained Adapter",
    'chat_template': "gemma-3",
    'base_model': BASE_MODEL_NAME  # For adapter loading
}

# Evaluation model
evaluator_config = bm_config.get('evaluator', {})
EVAL_API_CONFIG = {
    'base_url': evaluator_config.get('api_base_url', "http://localhost:11434/v1"),
    'api_key': evaluator_config.get('api_key', "ollama"),
    'model': evaluator_config.get('model', "gpt-oss:20b"),
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
# LANGUAGE DETECTION FOR EVALUATION
# =============================================================================

def detect_language(text: str) -> str:
    """Detects the language of the text (simple detection)."""
    # Simple language detection based on common words
    german_indicators = ['der', 'die', 'das', 'und', 'oder', 'ist', 'sind', 'von', 'mit', 'für', 'auf', 'in', 'zu', 'bei', 'nach', 'über', 'durch', 'unter', 'gegen', 'ohne']
    english_indicators = ['the', 'and', 'or', 'is', 'are', 'of', 'with', 'for', 'on', 'in', 'to', 'at', 'after', 'over', 'by', 'under', 'against', 'without']
    
    text_lower = text.lower()
    german_count = sum(1 for word in german_indicators if word in text_lower)
    english_count = sum(1 for word in english_indicators if word in text_lower)
    
    if german_count > english_count:
        return "German"
    elif english_count > german_count:
        return "English"
    else:
        return "German"  # Default to German for this pipeline

# =============================================================================
# INTELLIGENT MODEL DETECTION WITH FALLBACK SUPPORT
# =============================================================================

def detect_model_type(model_name: str, config_type: str = None) -> dict:
    """
    Detects model type and recommends optimal configuration with fallback info
    
    Args:
        model_name: Path to model
        config_type: Type from config ("merged", "adapter", "unknown", "base")
    """
    model_name_lower = model_name.lower()
    
    # Base model (PRE)
    if config_type == 'base':
        return {
            'type': 'base',
            'use_quantization': False,
            'use_unsloth': False,
            'dtype': torch.bfloat16 if 'gemma' in model_name_lower else torch.float16,
            'reason': 'Base model for PRE benchmark',
            'fallback_available': False,
            'fallback_adapter': None
        }
    
    # Use config type if available
    if config_type == 'adapter':
        # It's definitely an adapter
        adapter_path = model_name
        if adapter_path.endswith('_merged'):
            adapter_path = adapter_path.replace('_merged', '')
        
        # Check if adapter path exists
        if os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
            fallback_available = True
        else:
            fallback_available = False
            print(f"   ⚠️ Adapter path not found: {adapter_path}")
        
        return {
            'type': 'adapter',
            'use_quantization': False,
            'use_unsloth': False,
            'dtype': torch.float16,
            'reason': 'Adapter detected from config',
            'fallback_available': fallback_available,
            'fallback_adapter': adapter_path if fallback_available else None
        }
    
    # Check if it's an adapter directory
    if os.path.exists(model_name) and os.path.exists(os.path.join(model_name, 'adapter_config.json')):
        return {
            'type': 'adapter',
            'use_quantization': False,
            'use_unsloth': False,
            'dtype': torch.float16,
            'reason': 'Adapter directory detected',
            'fallback_available': True,
            'fallback_adapter': model_name
        }
    
    # Gemma models - special handling for cast errors
    if 'gemma' in model_name_lower:
        return {
            'type': 'gemma',
            'use_quantization': False,  # NO 4-bit for Gemma!
            'use_unsloth': False,
            'dtype': torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            'reason': 'Gemma detected - using bfloat16 without quantization',
            'fallback_available': False,
            'fallback_adapter': None
        }
    
    # Merged models
    if '_merged' in model_name_lower:
        # Check if adapter available for fallback
        adapter_path = find_adapter_for_merged(model_name)
        fallback_available = adapter_path is not None
        
        return {
            'type': 'merged',
            'use_quantization': False,
            'use_unsloth': False,
            'dtype': torch.float16,
            'reason': 'Merged model - no quantization needed',
            'fallback_available': fallback_available,
            'fallback_adapter': adapter_path
        }
    
    # Standard
    return {
        'type': 'standard',
        'use_quantization': True,
        'use_unsloth': UNSLOTH_AVAILABLE,
        'dtype': torch.float16,
        'reason': 'Standard model',
        'fallback_available': False,
        'fallback_adapter': None
    }

def ensure_model_downloaded(model_name):
    """Ensures the model is downloaded"""
    # Check if it's a local path
    if os.path.exists(model_name):
        print(f"✅ Local model found: {model_name}")
        return model_name
    
    # Check if it's a HuggingFace model
    if "/" in model_name and not model_name.startswith("./") and not model_name.startswith("../"):
        print(f"🔍 Checking HuggingFace model: {model_name}")
        
        # Try to download the model
        try:
            from huggingface_hub import snapshot_download
            
            print(f"📥 Downloading model: {model_name}")
            print("   This may take several minutes on first run...")
            
            # Download with token if available
            download_kwargs = {
                'repo_id': model_name,
                'cache_dir': cache_base,
            }
            
            if hf_token:
                download_kwargs['token'] = hf_token
            
            local_path = snapshot_download(**download_kwargs)
            print(f"✅ Model downloaded to: {local_path}")
            return local_path
            
        except Exception as e:
            print(f"❌ Download error: {e}")
            # Try with original name anyway
            return model_name
    
    return model_name

# =============================================================================
# MODEL WRAPPER with improved Device Management and AUTO-DOWNLOAD
# =============================================================================

class TransformersModelWrapper:
    """Wrapper with intelligent configuration, improved device management and fallback"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.chat_template = config.get('chat_template')
        # Pass config_type from pipeline config
        config_type = config.get('model_type', None)
        self.model_info = detect_model_type(config['model_name'], config_type)
        
        # If it's an adapter, go directly to adapter loading
        if self.model_info['type'] == 'adapter' and self.model_info.get('fallback_adapter'):
            # Set adapter path directly
            self.config['model_name'] = self.model_info['fallback_adapter']
            self.config['is_adapter'] = True
    
    def _load_base_with_adapter(self):
        """Fallback: Loads base model and applies adapter"""
        
        # Adapter path is already set correctly or needs to be determined
        adapter_path = self.config.get('model_name', '')
        
        # Remove _merged if present
        if adapter_path.endswith('_merged'):
            adapter_path = adapter_path.replace('_merged', '')
            print(f"   📂 Correcting path to: {adapter_path}")
        
        # Check if adapter exists
        if not os.path.exists(adapter_path):
            print(f"   ❌ Adapter path does not exist: {adapter_path}")
            return False
            
        if not os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
            print(f"   ❌ No valid adapter in: {adapter_path}")
            # Try searching in same directory
            parent = os.path.dirname(adapter_path)
            base_name = os.path.basename(adapter_path)
            
            # Try without _merged suffix
            if '_merged' in base_name:
                base_name = base_name.replace('_merged', '')
                adapter_path = os.path.join(parent, base_name)
                
                if not os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
                    print(f"   ❌ Also no adapter in: {adapter_path}")
                    return False
            else:
                return False
            
        print(f"   ✅ Using adapter: {adapter_path}")
            
        # Determine base model from config or adapter config
        base_model_name = self.config.get('base_model')
        if not base_model_name:
            base_model_name = get_base_model_for_adapter(adapter_path)
        
        # Ensure base model is downloaded
        base_model_path = ensure_model_downloaded(base_model_name)
        
        print(f"   🔍 Base model: {base_model_name}")
        print(f"   🔧 Adapter: {adapter_path}")
        
        try:
            # 2. Load base model tokenizer
            tokenizer_kwargs = {
                'trust_remote_code': True
            }
            
            if hf_token:
                tokenizer_kwargs['token'] = hf_token
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                **tokenizer_kwargs
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 3. Load base model
            model_kwargs = {
                'torch_dtype': self.model_info['dtype'],
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }
            
            # IMPORTANT: Explicit GPU device map
            if torch.cuda.is_available():
                free_vram = (torch.cuda.get_device_properties(0).total_memory - 
                           torch.cuda.memory_allocated()) / 1024**3
                if free_vram > 12:  # Enough VRAM for whole model
                    model_kwargs['device_map'] = {'': 0}  # Everything on GPU 0
                    print(f"   🎯 Forcing GPU-only loading (VRAM free: {free_vram:.1f} GB)")
                else:
                    model_kwargs['device_map'] = 'auto'
                    print(f"   ⚠️ Low VRAM ({free_vram:.1f} GB) - Auto device_map")
            else:
                model_kwargs['device_map'] = 'auto'
            
            if hf_token:
                model_kwargs['token'] = hf_token
            
            print("   🔧 WITHOUT quantization - Base model loading")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **model_kwargs
            )
            
            # 4. Load and apply adapter
            print("   🔧 Loading PEFT adapter...")
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                
                # 🚨 CRITICAL FIX: Ensure entire model is on GPU
                if torch.cuda.is_available():
                    print("   🔧 Moving adapter model completely to GPU...")
                    
                    # Method 1: Direct .cuda() call
                    try:
                        self.model = self.model.cuda()
                        print("   ✅ Model moved to GPU")
                    except Exception as cuda_error:
                        print(f"   ⚠️ .cuda() error: {cuda_error}")
                        
                        # Method 2: Manual device placement
                        try:
                            device = torch.device('cuda:0')
                            self.model = self.model.to(device)
                            print("   ✅ Model moved with .to(device)")
                        except Exception as to_error:
                            print(f"   ⚠️ .to(device) error: {to_error}")
                            return False
                
                # Check final device placement
                cpu_params = 0
                gpu_params = 0
                for name, param in self.model.named_parameters():
                    if param.device.type == 'cpu':
                        cpu_params += 1
                    else:
                        gpu_params += 1
                
                if cpu_params > 0:
                    print(f"   ⚠️ WARNING: {cpu_params} parameters still on CPU, {gpu_params} on GPU")
                else:
                    print(f"   ✅ All {gpu_params} parameters on GPU")
                
                print("   ✅ Base + Adapter successfully loaded!")
                return True
                
            except Exception as adapter_error:
                print(f"   ❌ PEFT Adapter error: {adapter_error}")
                return False
                
        except Exception as e:
            print(f"   ❌ Base+Adapter error: {e}")
            return False
    
    def load_model(self):
        """Loads model with optimal strategy, device map control and AUTO-DOWNLOAD"""
        model_name = os.path.basename(self.config['model_name'])
        print(f"\n🔄 Loading model: {model_name}")
        print(f"   📋 {self.model_info['reason']}")
        print(f"   📍 Full path: {self.config['model_name']}")
        
        # Cleanup before loading
        cleanup_memory()
        print_memory_status("   Before loading: ")
        
        # If already detected as adapter, load directly
        if self.config.get('is_adapter') or self.model_info['type'] == 'adapter':
            print("   🔧 Loading as adapter...")
            return self._load_base_with_adapter()
        
        # For base models: Ensure it's downloaded
        if self.model_info['type'] == 'base':
            print("   📥 Ensuring base model is available...")
            self.config['model_name'] = ensure_model_downloaded(self.config['model_name'])
        
        # Check if path exists
        model_path = self.config['model_name']
        
        if os.path.exists(model_path):
            # Local path exists
            print(f"   ✅ Local path exists: {model_path}")
            use_local = True
        else:
            # For HuggingFace models: Download if needed
            if "/" in model_path and not model_path.startswith("./") and not model_path.startswith("../"):
                print(f"   📥 Trying to load/download model: {model_path}")
                model_path = ensure_model_downloaded(model_path)
                use_local = True
            else:
                print(f"   ❌ Model not found: {model_path}")
                
                # Try fallback to adapter
                if self.model_info.get('fallback_available', False):
                    print("   🔄 Fallback: Trying base model + adapter...")
                    return self._load_base_with_adapter()
                else:
                    raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            # Try to load merged/base model
            return self._load_merged_model(model_path, use_local)
            
        except Exception as merged_error:
            error_msg = str(merged_error)
            print(f"   ⚠️ Model loading error: {error_msg[:200]}...")
            
            # Check fallback scenarios
            if self.model_info.get('fallback_available', False):
                print("   🔄 Fallback: Trying base model + adapter...")
                return self._load_base_with_adapter()
            
            elif "CUDA out of memory" in error_msg:
                print("   🚨 CUDA OOM - trying with quantization...")
                return self._try_quantization_fallback(model_path, use_local)
            
            else:
                print("   ❌ No suitable fallback available")
                raise merged_error
    
    def _try_quantization_fallback(self, model_path, use_local):
        """Fallback with quantization for memory problems"""
        print("   🔧 Quantization fallback activated")
        
        # Emergency Memory Recovery
        aggressive_cleanup()
        
        # Force quantization
        original_quantization = self.model_info['use_quantization']
        original_4bit = self.config.get('load_in_4bit', False)
        
        self.model_info['use_quantization'] = True
        self.config['load_in_4bit'] = True
        
        try:
            result = self._load_merged_model(model_path, use_local)
            print("   ✅ Quantization fallback successful!")
            return result
        except Exception as e:
            print(f"   ❌ Quantization fallback also failed: {e}")
            # Restore original settings
            self.model_info['use_quantization'] = original_quantization
            self.config['load_in_4bit'] = original_4bit
            return False
    
    def _load_merged_model(self, model_path, use_local):
        """Loads merged/base model (original logic with improvements)"""
        
        print(f"   📂 Loading from: {model_path}")
        
        # Load tokenizer
        tokenizer_kwargs = {
            'trust_remote_code': True
        }
        
        # Add token if available
        if hf_token:
            tokenizer_kwargs['token'] = hf_token
        
        # For local path always load from there
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                **tokenizer_kwargs
            )
        except Exception as e:
            print(f"   ⚠️ Tokenizer error: {e}")
            # Fallback: Try base model tokenizer
            if self.config.get('base_model'):
                print(f"   🔄 Loading base model tokenizer: {self.config['base_model']}")
                base_tokenizer_path = ensure_model_downloaded(self.config['base_model'])
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_tokenizer_path,
                    **tokenizer_kwargs
                )
            else:
                raise e
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model configuration with explicit device map
        model_kwargs = {
            'torch_dtype': self.model_info['dtype'],
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
        }
        
        # IMPORTANT: Explicit GPU device map if enough VRAM
        if torch.cuda.is_available():
            free_vram = (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_allocated()) / 1024**3
            if free_vram > 12:  # Enough VRAM for whole model
                model_kwargs['device_map'] = {'': 0}  # Everything on GPU 0
                print(f"   🎯 Forcing GPU-only loading (VRAM free: {free_vram:.1f} GB)")
            else:
                print(f"   ⚠️ Low VRAM ({free_vram:.1f} GB) - Auto device_map")
                model_kwargs['device_map'] = 'auto'
        else:
            model_kwargs['device_map'] = 'auto'
        
        # Add token if available
        if hf_token:
            model_kwargs['token'] = hf_token
        
        # Quantization only if recommended
        if self.model_info['use_quantization'] and self.config.get('load_in_4bit'):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.model_info['dtype'],
                bnb_4bit_use_double_quant=True
            )
            model_kwargs['quantization_config'] = bnb_config
            print("   🔧 With 4-bit quantization")
        else:
            print(f"   🔧 WITHOUT quantization - {self.model_info['dtype']}")
        
        # Load model - directly from local path
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            raise e
        
        # Check device placement
        is_fully_on_gpu = check_device_placement(self.model)
        if not is_fully_on_gpu:
            print("   ⚠️ WARNING: Model partially on CPU - performance will suffer!")
        
        print(f"   ✅ Model loaded")
        print_memory_status("   After loading: ")
        
        return True
    
    def generate_answer(self, question: str, question_id: str) -> str:
        """Generates answer with device error workaround"""
        if not self.model or not self.tokenizer:
            return "Error: Model not loaded"
        
        try:
            # Chat template
            if self.chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": question}]
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = f"Question: {question}\nAnswer:"
            
            # Tokenization
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config['max_seq_length'] - INFERENCE_CONFIG['max_new_tokens'],
            )
            
            # 🚨 DEVICE-FIX: Ensure inputs are on same device as model
            if torch.cuda.is_available():
                # Determine model device
                try:
                    model_device = next(self.model.parameters()).device
                    # Move inputs to same device
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                except:
                    # Fallback: Try CUDA
                    inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generation with improved error handling
            with torch.no_grad():
                try:
                    if 'gemma' in self.config['model_name'].lower():
                        # Special handling for Gemma
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
                        print(f"   ⚠️ Device error: Attempting repair...")
                        # Try moving all model parameters to GPU
                        if torch.cuda.is_available():
                            try:
                                self.model = self.model.cuda()
                                inputs = {k: v.cuda() for k, v in inputs.items()}
                            except:
                                pass
                        
                        # Simplified second attempt
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=INFERENCE_CONFIG['max_new_tokens'],
                            temperature=0.7,
                            do_sample=False,  # Simplified generation
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    else:
                        raise device_error
            
            # Decoding
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return answer.strip()
            
        except Exception as e:
            print(f"   ⚠️ Generation error: {str(e)[:100]}")
            return "Answer could not be generated."
    
    def unload_model(self):
        """Unloads model completely"""
        print("   🗑️ Unloading model...")
        if self.model:
            # Try moving to CPU before deleting
            try:
                self.model.cpu()
            except:
                pass
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Aggressive cleanup
        aggressive_cleanup()
        print("   ✅ Model unloaded and memory cleaned")

# =============================================================================
# BENCHMARK CLASS with BATCH MODE
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
        
        print(f"🔧 Benchmark initialized")
        print(f"   📝 Model: {os.path.basename(answer_config['model_name'])}")
        print(f"   ⚖️ Evaluator: {'API' if use_api_eval else 'Local'}")
    
    def load_questions(self, file_path: str = None):
        """Loads questions"""
        if file_path is None:
            file_path = bm_config.get('questions_file', "BENCHMARKFRAGEN/benchmark_fragen_complete.json")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.questions_data = json.load(f)
            
            total = sum(len(k['fragen']) for k in self.questions_data['kategorien'])
            print(f"✅ {total} questions loaded from {file_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading: {e}")
            return False
    
    def evaluate_answer(self, question: str, generated: str, correct: str, question_id: str):
        """Evaluates answer with language-agnostic prompt"""
        try:
            # Detect language of the content for appropriate evaluation
            content_language = detect_language(question + " " + correct)
            
            # Create language-agnostic evaluation prompt
            if content_language == "German":
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
            else:
                prompt = f"""Evaluate this answer on a scale of 0-10.

QUESTION: {question}

REFERENCE SOLUTION: {correct}

GIVEN ANSWER: {generated}

Evaluate based on:
1. Completeness (0-2)
2. Correctness (0-2)
3. Precision (0-2)
4. No hallucinations (0-2)
5. Structure (0-2)

Return only the TOTAL POINTS as a number between 0 and 10."""
            
            if self.use_api_eval and self.eval_client:
                response = self.eval_client.chat.completions.create(
                    model=self.eval_config['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100
                )
                
                text = response.choices[0].message.content.strip()
                
                # Extract score
                numbers = re.findall(r'\d+', text)
                if numbers:
                    score = int(numbers[-1])  # Take the last number
                    score = max(0, min(10, score))
                else:
                    score = 5
                
                return score, text
            else:
                return 5, "Local evaluation not implemented"
                
        except Exception as e:
            print(f"   ⚠️ Evaluation error: {e}")
            return 0, "Error"
    
    def _generate_answers_only(self, description):
        """Generates only answers without evaluation"""
        print(f"\n🚀 Generating {description} answers")
        
        # Load model
        self.answer_model = TransformersModelWrapper(self.answer_config)
        if not self.answer_model.load_model():
            print(f"❌ {description} model could not be loaded")
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
        
        # Unload model immediately
        self.answer_model.unload_model()
        print(f"✅ {description} model unloaded")
        
        return answers
    
    def _evaluate_answers_batch(self, answers):
        """Evaluates all answers in a batch"""
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
        """Calculates percentage score"""
        scores = [r['score'] for r in results]
        return (sum(scores) / (len(scores) * 10)) * 100 if scores else 0
    
    def _show_results(self, results):
        """Shows final results"""
        if 'pre' in results and 'post' in results:
            pre_score = results['pre']['percentage_score']
            post_score = results['post']['percentage_score']
            improvement = post_score - pre_score
            
            print(f"\n🏆 COMPLETE RESULTS:")
            print(f"   Pre: {pre_score:.1f}%")
            print(f"   Post: {post_score:.1f}%")
            print(f"   📈 Improvement: {improvement:+.1f}%")
            
            if improvement > 0:
                print("   ✅ Finetuning successful!")
            else:
                print("   ⚠️ No measurable improvement")
                
        elif 'post' in results:
            post_score = results['post']['percentage_score']
            print(f"\n⚠️ PARTIAL RESULTS (POST only):")
            print(f"   Post: {post_score:.1f}%")
            
        elif 'pre' in results:
            pre_score = results['pre']['percentage_score']
            print(f"\n⚠️ PARTIAL RESULTS (PRE only):")
            print(f"   Pre: {pre_score:.1f}%")
    
    def run_benchmark(self, description=""):
        """Runs single benchmark"""
        if not self.questions_data:
            print("❌ No questions loaded")
            return None
        
        print(f"\n🚀 Starting benchmark {description}")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # Phase 1: Generate answers
        print("\n📝 Phase 1: Generate answers")
        answers = self._generate_answers_only(description)
        
        if not answers:
            return None
        
        # Phase 2: Evaluate
        print(f"\n⚖️ Phase 2: Evaluate")
        self.results = self._evaluate_answers_batch(answers)
        
        # Statistics
        duration = datetime.now() - start_time
        percentage = self._calculate_score(self.results)
        scores = [r['score'] for r in self.results]
        
        print(f"\n\n✅ FINISHED")
        print(f"   ⏱️ {duration}")
        print(f"   📊 Score: {percentage:.1f}%")
        print(f"   📈 Average: {statistics.mean(scores):.1f}/10")
        
        return {
            'config': self.answer_config,
            'results': self.results,
            'percentage_score': percentage,
            'duration': duration
        }
    
    def run_comparison(self, pre_config, post_config):
        """Comparison benchmark with BATCH MODE to avoid VRAM conflicts"""
        print("🎯 COMPARISON BENCHMARK - BATCH MODE")
        print("=" * 60)
        
        all_answers = {}  # Collects all answers
        results = {}
        
        # ========================================
        # PHASE 1: GENERATE ALL ANSWERS
        # ========================================
        print("\n📝 PHASE 1: GENERATE ANSWERS")
        print("=" * 40)
        print("🎯 Generate all answers WITHOUT Ollama evaluator")
        print("   (Avoids VRAM conflict between models)")
        
        # Pre-finetuning answers
        print("\n🔄 Generating PRE-finetuning answers...")
        print(f"   Base model: {pre_config['model_name']}")
        print_memory_status("Before pre-generation: ")
        
        pre_benchmark = TransformersBenchmark(pre_config, self.eval_config, use_api_eval=False)
        if pre_benchmark.load_questions():
            pre_answers = pre_benchmark._generate_answers_only("PRE")
            all_answers['pre'] = pre_answers
            print(f"✅ {len(pre_answers)} PRE answers generated")
        
        # Aggressive cleanup between models
        ultra_cleanup("Pre-Generation")
        
        # Post-finetuning answers  
        print("\n🔄 Generating POST-finetuning answers...")
        print(f"   Trained model: {post_config['model_name']}")
        print_memory_status("Before post-generation: ")
        
        post_benchmark = TransformersBenchmark(post_config, self.eval_config, use_api_eval=False)
        if post_benchmark.load_questions():
            post_answers = post_benchmark._generate_answers_only("POST")
            all_answers['post'] = post_answers
            print(f"✅ {len(post_answers)} POST answers generated")
        
        # Final cleanup before evaluation
        ultra_cleanup("Post-Generation")
        
        # ========================================
        # PHASE 2: EVALUATE ALL ANSWERS
        # ========================================
        print("\n⚖️ PHASE 2: BATCH EVALUATION")
        print("=" * 40)
        print("🤖 Starting Ollama evaluator...")
        print("   (Now VRAM is free for evaluation)")
        
        # Wait until enough VRAM is free
        if not wait_for_vram_free(min_gb=12):
            print("❌ Not enough VRAM for Ollama evaluator")
            return results
        
        # Now VRAM is free for Ollama
        evaluator = TransformersBenchmark(pre_config, self.eval_config, use_api_eval=True)
        
        # Evaluate all PRE answers
        if 'pre' in all_answers:
            print("\n📊 Evaluating PRE answers...")
            pre_results = evaluator._evaluate_answers_batch(all_answers['pre'])
            results['pre'] = {
                'config': pre_config,
                'results': pre_results,
                'percentage_score': evaluator._calculate_score(pre_results),
                'duration': None
            }
        
        # Evaluate all POST answers
        if 'post' in all_answers:
            print("\n📊 Evaluating POST answers...")
            post_results = evaluator._evaluate_answers_batch(all_answers['post'])
            results['post'] = {
                'config': post_config,
                'results': post_results,
                'percentage_score': evaluator._calculate_score(post_results),
                'duration': None
            }
        
        # ========================================
        # RESULTS
        # ========================================
        evaluator._show_results(results)
        
        # Save results
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
            
            print(f"\n💾 Saved: {filename}")
        
        return results

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function with central configuration and batch mode"""
    print("🎯 OTW Benchmark Tool - WITH AUTO-DOWNLOAD + DYNAMIC CONFIG")
    print("✅ Base model from finetuning config")
    print("✅ Automatic download when needed")
    print("✅ Batch mode against VRAM conflicts")
    print("✅ Base + Adapter fallback")
    print("✅ Ollama-compatible memory management")
    print("✅ Language-agnostic evaluation")
    print("=" * 60)
    
    # Immediate startup cleanup
    print("\n🧹 Startup memory cleanup...")
    aggressive_cleanup()
    
    # Show cache status
    print(f"\n📍 Cache directory: {cache_base}")
    print_memory_status("Start status: ")
    
    # Show used models
    print(f"\n📋 Configured models:")
    print(f"   Base model: {BASE_MODEL_NAME}")
    print(f"   Trained model: {POST_FINETUNING_CONFIG['model_name']}")
    print(f"   Type: {POST_FINETUNING_CONFIG['model_type']}")
    
    # Use evaluator type from config
    use_api = evaluator_config.get('type', 'api') == 'api'
    
    # Benchmark based on mode from config
    if BENCHMARK_MODE == "comparison":
        # Comparison with batch mode
        print(f"\n🔄 Starting batch comparison benchmark")
        print(f"   Pre-Model: {PRE_FINETUNING_CONFIG['model_name']}")
        print(f"   Post-Model: {POST_FINETUNING_CONFIG['model_name']}")
        print(f"   🎯 BATCH MODE activated:")
        print(f"      1. Generate PRE answers → Unload model")
        print(f"      2. Generate POST answers → Unload model") 
        print(f"      3. Start Ollama evaluator → Evaluate ALL")
        print(f"      ✅ No VRAM conflict between models!")
        
        benchmark = TransformersBenchmark(
            PRE_FINETUNING_CONFIG,
            EVAL_API_CONFIG,
            use_api
        )
        if benchmark.load_questions():
            benchmark.run_comparison(PRE_FINETUNING_CONFIG, POST_FINETUNING_CONFIG)
            
    elif BENCHMARK_MODE == "post_only":
        # Post only
        benchmark = TransformersBenchmark(
            POST_FINETUNING_CONFIG,
            EVAL_API_CONFIG,
            use_api
        )
        if benchmark.load_questions():
            print(f"\n📝 Starting post benchmark")
            print(f"   Model: {POST_FINETUNING_CONFIG['model_name']}")
            benchmark.run_benchmark("POST-FINETUNING")
            
    elif BENCHMARK_MODE == "pre_only":
        # Pre only
        benchmark = TransformersBenchmark(
            PRE_FINETUNING_CONFIG,
            EVAL_API_CONFIG,
            use_api
        )
        if benchmark.load_questions():
            print(f"\n📝 Starting pre benchmark")
            print(f"   Model: {PRE_FINETUNING_CONFIG['model_name']}")
            benchmark.run_benchmark("PRE-FINETUNING")
            
    else:
        print(f"❌ Unknown benchmark mode: {BENCHMARK_MODE}")
        print("   Valid modes: comparison, post_only, pre_only")
        return
    
    # Final cleanup
    aggressive_cleanup()
    print_memory_status("\nEnd status: ")
    
    # Re-enable PyTorch Compile
    os.environ.pop('TORCH_COMPILE_DISABLE', None)
    print("\n✅ Finished - PyTorch Compile re-enabled")

if __name__ == "__main__":
    main()