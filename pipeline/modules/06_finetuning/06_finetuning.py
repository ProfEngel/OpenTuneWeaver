# -*- coding: utf-8 -*-
"""
Gemma 3N Finetuning Script - Mit zentraler Konfiguration
Fine-tuning script using central pipeline_config.json
"""

import os
import sys
import time
import torch
import json
from datetime import datetime
from pathlib import Path

# ========================================
# ZENTRALE KONFIGURATION LADEN
# ========================================
sys.path.append(str(Path(__file__).parent.parent.parent))  # Zum Hauptverzeichnis
from config_loader import PipelineConfigLoader

# Lade Konfiguration für dieses Modul
config_loader = PipelineConfigLoader()
ft_config = config_loader.get_finetuning_config()
tokens = config_loader.get_tokens()

# Zeige geladene Konfiguration
print("=" * 60)
print("📋 KONFIGURATION GELADEN (06_finetuning)")
print("=" * 60)
config_loader.print_config_summary()
print(f"\n  🎯 Modell: {ft_config.get('model_name', 'Unknown')}")
print(f"  🤖 Base: {ft_config.get('base_model', 'Unknown')}")
print(f"  📊 Profil: {ft_config.get('training_profile', 'Unknown')}")
print(f"  🔄 Epochs: {ft_config.get('num_train_epochs', 0)}")
print(f"  💾 Merged: {'✅' if ft_config.get('save_merged') else '❌'}")
print(f"  🎯 GGUF: {'✅' if ft_config.get('save_gguf') else '❌'}")
print(f"  🔑 HF-Token: {'✅' if tokens.get('hf_token') else '❌'}")
print("=" * 60)

# Setze Environment-Variablen für HF-Tokens
if tokens.get('hf_token'):
    os.environ["HF_TOKEN"] = tokens['hf_token']
    os.environ["HUGGINGFACE_TOKEN"] = tokens['hf_token']
if tokens.get('hf_write_token'):
    os.environ["HF_WRITE_TOKEN"] = tokens['hf_write_token']

# Import unsloth
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi

# Configuration Class mit Werten aus zentraler Config
class Config:
    # Aus zentraler Config laden
    MODEL_NAME = ft_config.get('base_model', 'unsloth/gemma-3n-E2B-it')
    OUTPUT_MODEL_NAME = ft_config.get('model_name', 'FoxLM-e2b')
    HF_REPO_ID = ft_config.get('hf_repo_id', f"user/{ft_config.get('model_name', 'model')}")
    DATASET_PATH = ft_config.get('dataset_path', 'INPUT/dataset.json')
    CHAT_TEMPLATE = ft_config.get('chat_template', 'gemma-3')
    CUSTOM_MODEL_DIR = ft_config.get('custom_model_dir', 'CustomModel')
    
    # Training settings
    MAX_SEQ_LENGTH = ft_config.get('max_seq_length', 8192)
    LOAD_IN_4BIT = ft_config.get('load_in_4bit', True)
    FULL_FINETUNING = ft_config.get('full_finetuning', False)
    
    # LoRA settings
    LORA_R = ft_config.get('lora_r', 8)
    LORA_ALPHA = ft_config.get('lora_alpha', 8)
    LORA_DROPOUT = ft_config.get('lora_dropout', 0)
    BIAS = ft_config.get('bias', 'none')
    RANDOM_STATE = ft_config.get('random_state', 3407)
    
    # Training parameters
    PER_DEVICE_TRAIN_BATCH_SIZE = ft_config.get('per_device_train_batch_size', 1)
    GRADIENT_ACCUMULATION_STEPS = ft_config.get('gradient_accumulation_steps', 16)
    WARMUP_STEPS = ft_config.get('warmup_steps', 200)
    NUM_TRAIN_EPOCHS = ft_config.get('num_train_epochs', 15)
    MAX_STEPS = ft_config.get('max_steps', -1)
    LEARNING_RATE = ft_config.get('learning_rate', 5e-5)
    LOGGING_STEPS = ft_config.get('logging_steps', 5)
    OPTIM = ft_config.get('optim', 'adamw_8bit')
    WEIGHT_DECAY = ft_config.get('weight_decay', 0.03)
    LR_SCHEDULER_TYPE = ft_config.get('lr_scheduler_type', 'cosine')
    SEED = ft_config.get('seed', 3407)
    
    # Output settings
    SAVE_LOCAL = True
    SAVE_LORA = ft_config.get('save_lora', True)
    SAVE_MERGED = ft_config.get('save_merged', True)
    SAVE_GGUF = ft_config.get('save_gguf', True)
    UPLOAD_TO_HF = ft_config.get('upload_to_hf', False)
    GGUF_QUANTIZATIONS = ft_config.get('gguf_quantizations', ['q8_0'])
    
    # Inference settings
    TEMPERATURE = ft_config.get('temperature', 1.0)
    TOP_P = ft_config.get('top_p', 0.95)
    TOP_K = ft_config.get('top_k', 64)
    MAX_NEW_TOKENS = ft_config.get('max_new_tokens', 128)
    
    # Tokens
    HF_TOKEN = tokens.get('hf_token')
    HF_WRITE_TOKEN = tokens.get('hf_write_token', tokens.get('hf_token'))
    
    # Auto-optimization
    AUTO_OPTIMIZE = True
    DETECTED_GPU_MEMORY = 0
    DETECTED_GPU_NAME = ""

def check_model_in_cache(model_name):
    """Check if model is already in cache"""
    cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    if not os.path.exists(cache_dir):
        return False
    
    cache_model_name = model_name.replace("/", "--")
    cache_entries = [d for d in os.listdir(cache_dir) if cache_model_name in d and os.path.isdir(os.path.join(cache_dir, d))]
    
    if cache_entries:
        cache_path = os.path.join(cache_dir, cache_entries[0])
        snapshots_dir = os.path.join(cache_path, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshot_dirs = os.listdir(snapshots_dir)
            if snapshot_dirs:
                latest_snapshot = os.path.join(snapshots_dir, snapshot_dirs[0])
                model_files = [f for f in os.listdir(latest_snapshot) if f.endswith('.safetensors')]
                if model_files:
                    total_size = sum(os.path.getsize(os.path.join(latest_snapshot, f)) for f in model_files) / (1024**3)
                    print(f"✅ Model bereits im Cache: {model_name} ({total_size:.1f} GB)")
                    return True
    
    print(f"📥 Model wird beim ersten Laden heruntergeladen: {model_name}")
    return False

def detect_gpu_and_optimize(config):
    """Automatically detect GPU and optimize settings"""
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. Using CPU (very slow).")
        config.LOAD_IN_4BIT = False
        return config
    
    # PyTorch optimizations
    try:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 512
        torch._dynamo.config.suppress_errors = True
        print("🔧 Applied PyTorch Dynamo fixes")
    except Exception as e:
        print(f"⚠️ Could not apply Dynamo fixes: {e}")
    
    try:
        torch._dynamo.reset()
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        print("🔧 Reset PyTorch optimizations")
    except Exception as e:
        print(f"⚠️ Could not reset PyTorch optimizations: {e}")
    
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_name = gpu_props.name
    gpu_memory_gb = round(gpu_props.total_memory / 1024 / 1024 / 1024, 1)
    
    config.DETECTED_GPU_NAME = gpu_name
    config.DETECTED_GPU_MEMORY = gpu_memory_gb
    
    print(f"🎯 Detected GPU: {gpu_name} ({gpu_memory_gb} GB VRAM)")
    
    # Enable GPU-specific optimizations
    torch.backends.cudnn.benchmark = True
    
    # TF32 optimizations
    if any(x in gpu_name.upper() for x in ['A4000', 'A4500', 'A5000', 'A6000', 'RTX 30', 'RTX 40', 'RTX 3', 'RTX 4']):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ Enabled TF32 optimizations (Ampere+ GPU detected)")
    
    # Determine if using 2B or 4B model
    is_4b_model = "4B" in config.MODEL_NAME.upper() or "E4B" in config.MODEL_NAME.upper()
    model_size = "4B" if is_4b_model else "2B"
    print(f"📊 Model size: {model_size}")
    
    # Auto-adjust settings based on GPU memory
    if gpu_memory_gb >= 20:
        if is_4b_model:
            config.PER_DEVICE_TRAIN_BATCH_SIZE = 2
            config.GRADIENT_ACCUMULATION_STEPS = 8
            config.MAX_SEQ_LENGTH = min(config.MAX_SEQ_LENGTH, 6144)
        else:
            config.PER_DEVICE_TRAIN_BATCH_SIZE = 3
            config.GRADIENT_ACCUMULATION_STEPS = 6
            config.MAX_SEQ_LENGTH = min(config.MAX_SEQ_LENGTH, 8192)
    elif gpu_memory_gb >= 16:
        if is_4b_model:
            config.PER_DEVICE_TRAIN_BATCH_SIZE = 1
            config.GRADIENT_ACCUMULATION_STEPS = 16
            config.MAX_SEQ_LENGTH = min(config.MAX_SEQ_LENGTH, 4096)
        else:
            config.PER_DEVICE_TRAIN_BATCH_SIZE = 2
            config.GRADIENT_ACCUMULATION_STEPS = 8
            config.MAX_SEQ_LENGTH = min(config.MAX_SEQ_LENGTH, 6144)
    elif gpu_memory_gb >= 12:
        config.PER_DEVICE_TRAIN_BATCH_SIZE = 1
        config.GRADIENT_ACCUMULATION_STEPS = 16
        config.MAX_SEQ_LENGTH = min(config.MAX_SEQ_LENGTH, 4096)
    elif gpu_memory_gb >= 8:
        if is_4b_model:
            print("⚠️ 4B model may be too large for 8GB VRAM. Consider using 2B model.")
        config.PER_DEVICE_TRAIN_BATCH_SIZE = 1
        config.GRADIENT_ACCUMULATION_STEPS = 16 if not is_4b_model else 32
        config.MAX_SEQ_LENGTH = min(config.MAX_SEQ_LENGTH, 4096)
    else:
        if is_4b_model:
            print("❌ 4B model not recommended for low VRAM. Switching to 2B model.")
            config.MODEL_NAME = config.MODEL_NAME.replace("E4B", "E2B").replace("4B", "2B")
        config.PER_DEVICE_TRAIN_BATCH_SIZE = 1
        config.GRADIENT_ACCUMULATION_STEPS = 32
        config.MAX_SEQ_LENGTH = min(config.MAX_SEQ_LENGTH, 4096)
    
    effective_batch_size = config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS
    print(f"📈 Effective batch size: {effective_batch_size}")
    print(f"📏 Max sequence length: {config.MAX_SEQ_LENGTH}")
    
    return config

def load_model_and_tokenizer(config):
    """Load the base model and tokenizer"""
    print(f"Loading model: {config.MODEL_NAME}")
    
    if not config.HF_TOKEN:
        print("⚠️ No HF_TOKEN found. Make sure you're logged in with 'huggingface-cli login'")
    
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.MODEL_NAME,
            dtype=None,
            max_seq_length=config.MAX_SEQ_LENGTH,
            load_in_4bit=config.LOAD_IN_4BIT,
            full_finetuning=config.FULL_FINETUNING,
            token=config.HF_TOKEN,
        )
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\n🔧 Authentication error. Please check HF_TOKEN in pipeline_config.json")
        raise e

def setup_lora(model, config):
    """Setup LoRA adapters"""
    print("Setting up LoRA adapters...")
    
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias=config.BIAS,
        random_state=config.RANDOM_STATE,
    )
    
    print("LoRA adapters configured!")
    return model

def prepare_dataset(config, tokenizer):
    """Load and prepare dataset"""
    print(f"Loading dataset from: {config.DATASET_PATH}")
    
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {config.DATASET_PATH}")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=config.DATASET_PATH, split="train")
        
        print(f"Dataset size: {len(dataset)} examples")
        
        # Setup chat template
        tokenizer = get_chat_template(tokenizer, chat_template=config.CHAT_TEMPLATE)
        
        # Format dataset
        def format_for_sft(example):
            try:
                if 'conversations' in example:
                    conversations = example["conversations"]
                elif 'messages' in example:
                    conversations = example["messages"]
                else:
                    conversations = example
                
                formatted_text = tokenizer.apply_chat_template(
                    conversations,
                    tokenize=False,
                    add_generation_prompt=False
                )
                
                # Remove BOS token if present
                if formatted_text.startswith('<bos>'):
                    formatted_text = formatted_text[5:]
                elif formatted_text.startswith('<s>'):
                    formatted_text = formatted_text[4:]
                
                return {"text": formatted_text}
            except Exception as e:
                print(f"⚠️ Error formatting conversation: {e}")
                return {"text": str(conversations)}
        
        dataset = dataset.map(format_for_sft, remove_columns=dataset.column_names)
        print("✅ Dataset prepared successfully!")
        
        return dataset, tokenizer
        
    except Exception as e:
        print(f"❌ Error preparing dataset: {e}")
        raise e

def setup_trainer(model, tokenizer, dataset, config):
    """Setup the SFT trainer"""
    import torch
    print("Setting up trainer...")
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("🔧 Set pad_token to eos_token")
    
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    
    # Data collator
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    # Trainer configuration
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=config.WARMUP_STEPS,
            num_train_epochs=config.NUM_TRAIN_EPOCHS,
            max_steps=config.MAX_STEPS,
            learning_rate=config.LEARNING_RATE,
            logging_steps=config.LOGGING_STEPS,
            optim=config.OPTIM,
            weight_decay=config.WEIGHT_DECAY,
            lr_scheduler_type=config.LR_SCHEDULER_TYPE,
            seed=config.SEED,
            report_to="none",
            output_dir=f"./results/{config.CUSTOM_MODEL_DIR}/{config.OUTPUT_MODEL_NAME}",
            
            # Critical settings
            max_seq_length=config.MAX_SEQ_LENGTH,
            packing=False,
            remove_unused_columns=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            
            # Precision settings
            bf16=True if torch.cuda.is_bf16_supported() else False,
            fp16=False if torch.cuda.is_bf16_supported() else True,
            gradient_checkpointing=True,
            
            # Memory optimizations
            group_by_length=False,
            dataloader_drop_last=True,
        ),
    )
    
    # Apply response-only training
    trainer = train_on_responses_only(
        trainer,
        instruction_part="user\n",
        response_part="model\n",
    )
    
    print("✅ Trainer configured!")
    return trainer

def train_model(trainer, config):
    """Train the model"""
    print("🚀 Starting training...")
    
    start_memory, max_memory = show_memory_stats(config)
    trainer_stats = trainer.train()
    
    # Show final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    print(f"\nTraining completed!")
    print(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    print(f"Peak memory usage: {used_memory} GB ({used_percentage}%)")
    print(f"Memory for LoRA: {used_memory_for_lora} GB ({lora_percentage}%)")
    
    return trainer_stats

def show_memory_stats(config=None):
    """Display GPU memory statistics"""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        memory_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        
        if config and config.DETECTED_GPU_NAME:
            print(f"🎯 GPU: {config.DETECTED_GPU_NAME} ({config.DETECTED_GPU_MEMORY} GB)")
        else:
            print(f"GPU: {gpu_stats.name} (Max memory: {max_memory} GB)")
        
        print(f"📊 Current memory usage: {memory_reserved} GB")
        return memory_reserved, max_memory
    else:
        print("💻 Running on CPU")
        return 0, 0

def test_inference(model, tokenizer, config):
    """Test the trained model"""
    print("\nTesting inference...")
    
    # Setup tokenizer for inference
    tokenizer = get_chat_template(tokenizer, chat_template=config.CHAT_TEMPLATE)
    
    # Test question
    test_message = "Was ist der Sinn des Lebens?"
    
    print(f"\n--- Inference Test ---")
    print(f"User: {test_message}")
    print("Assistant: ", end="")
    
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": test_message}]
    }]
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        _ = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )
        
        print("\n✅ Inference test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in inference test: {e}")

def save_model(model, tokenizer, config):
    """Save the trained model locally"""
    if not config.SAVE_LOCAL:
        print("🚫 Lokale Speicherung übersprungen")
        return
    
    # Create directory
    custom_model_path = os.path.join(config.CUSTOM_MODEL_DIR, config.OUTPUT_MODEL_NAME)
    os.makedirs(config.CUSTOM_MODEL_DIR, exist_ok=True)
    os.makedirs(custom_model_path, exist_ok=True)
    
    print(f"\n💾 Speichere Modell: {custom_model_path}")
    
    saved_formats = []
    
    try:
        # Save LoRA adapters
        if config.SAVE_LORA:
            print("📌 Speichere LoRA Adapter...")
            model.save_pretrained(custom_model_path)
            tokenizer.save_pretrained(custom_model_path)
            saved_formats.append("LoRA Adapter")
            print("✅ LoRA Adapter gespeichert!")
        
        # Save merged model
        if config.SAVE_MERGED:
            print("🔗 Speichere Merged Model...")
            try:
                merged_path = f"{custom_model_path}_merged"
                model.save_pretrained_merged(
                    merged_path,
                    tokenizer,
                    save_method="merged_16bit",
                    maximum_memory_usage=0.8,
                    token=config.HF_TOKEN,
                    safe_serialization=True
                )
                saved_formats.append("Merged Model (16bit)")
                print("✅ Merged Model gespeichert!")
                
                # GGUF conversion if requested
                if config.SAVE_GGUF and config.GGUF_QUANTIZATIONS:
                    print(f"🎯 Erstelle GGUF-Dateien: {', '.join(config.GGUF_QUANTIZATIONS)}")
                    
                    for quantization in config.GGUF_QUANTIZATIONS:
                        print(f"   🔄 Konvertiere zu {quantization}...")
                        
                        # Find convert script
                        convert_script_paths = [
                            "llama.cpp/convert-hf-to-gguf.py",
                            "llama.cpp/convert_hf_to_gguf.py",
                            "convert-hf-to-gguf.py",
                            "convert_hf_to_gguf.py"
                        ]
                        
                        convert_script = None
                        for path in convert_script_paths:
                            if os.path.exists(path):
                                convert_script = path
                                break
                        
                        if convert_script:
                            try:
                                import subprocess
                                output_file = os.path.join(config.CUSTOM_MODEL_DIR, f"{config.OUTPUT_MODEL_NAME}-{quantization}.gguf")
                                
                                cmd = [
                                    sys.executable,
                                    convert_script,
                                    merged_path,
                                    "--outfile", output_file,
                                    "--outtype", quantization
                                ]
                                
                                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                                
                                if result.returncode == 0:
                                    if os.path.exists(output_file):
                                        size_gb = os.path.getsize(output_file) / (1024**3)
                                        print(f"   ✅ {quantization}: {os.path.basename(output_file)} ({size_gb:.1f} GB)")
                                        saved_formats.append(f"GGUF {quantization}")
                                else:
                                    print(f"   ❌ {quantization}: Konvertierung fehlgeschlagen")
                                    
                            except Exception as e:
                                print(f"   ❌ {quantization} Fehler: {e}")
                        else:
                            print(f"   ❌ convert-hf-to-gguf.py nicht gefunden")
                            print("   💡 Installiere llama.cpp für GGUF-Konvertierung")
                
            except Exception as e:
                print(f"⚠️ Merged model save failed: {e}")
        
        # Summary
        print(f"\n✅ Lokale Speicherung abgeschlossen!")
        print(f"📁 Speicherort: {config.CUSTOM_MODEL_DIR}/")
        print(f"📋 Gespeicherte Formate:")
        for fmt in saved_formats:
            print(f"   • {fmt}")
        
        # Create model info file
        create_model_info_file(config, saved_formats)
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        raise e

def create_model_info_file(config, saved_formats):
    """Create info file for benchmark script"""
    model_info = {
        'model_name': config.OUTPUT_MODEL_NAME,
        'model_path': os.path.join(config.CUSTOM_MODEL_DIR, config.OUTPUT_MODEL_NAME),
        'base_model': config.MODEL_NAME,
        'chat_template': config.CHAT_TEMPLATE,
        'created_timestamp': datetime.now().isoformat(),
        'saved_formats': saved_formats,
        'config': {
            'max_seq_length': config.MAX_SEQ_LENGTH,
            'load_in_4bit': config.LOAD_IN_4BIT,
            'epochs': config.NUM_TRAIN_EPOCHS,
            'learning_rate': config.LEARNING_RATE
        }
    }
    
    # Save info file
    info_file = os.path.join(config.CUSTOM_MODEL_DIR, f"{config.OUTPUT_MODEL_NAME}_info.json")
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Modell-Info gespeichert: {info_file}")
    
    # Create global info file
    global_info_file = "latest_model_info.json"
    with open(global_info_file, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Globale Modell-Info erstellt: {global_info_file}")

def upload_to_hub(model, tokenizer, config):
    """Upload model to Hugging Face Hub"""
    if not config.UPLOAD_TO_HF:
        print("🚫 HF Upload übersprungen")
        return
    
    upload_token = config.HF_WRITE_TOKEN or config.HF_TOKEN
    
    if not upload_token:
        print("❌ Kein HF-Token verfügbar. Upload übersprungen.")
        return
    
    print(f"\n☁️ Uploade zu Hugging Face Hub: {config.HF_REPO_ID}")
    
    try:
        # Test authentication
        from huggingface_hub import whoami
        try:
            user_info = whoami(token=upload_token)
            print(f"✅ Authenticated as: {user_info['name']}")
        except Exception as auth_error:
            print(f"❌ Authentication failed: {auth_error}")
            return
        
        # Upload LoRA adapters
        if config.SAVE_LORA:
            print("📌 Uploade LoRA Adapter...")
            model.push_to_hub(config.HF_REPO_ID, token=upload_token)
            tokenizer.push_to_hub(config.HF_REPO_ID, token=upload_token)
            print("✅ LoRA adapters uploaded!")
        
        # Upload merged model
        if config.SAVE_MERGED:
            try:
                print("🔗 Uploade Merged Model...")
                model.push_to_hub_merged(
                    config.HF_REPO_ID,
                    tokenizer,
                    token=upload_token,
                    maximum_memory_usage=0.8
                )
                print("✅ Merged model uploaded!")
            except Exception as merged_error:
                print(f"⚠️ Merged model upload failed: {merged_error}")
        
        print("✅ Upload zu Hub erfolgreich abgeschlossen!")
        
    except Exception as e:
        print(f"❌ Fehler beim Upload zu Hub: {e}")

def main():
    """Main function"""
    print("🎯 Gemma 3N Fine-tuning - MIT ZENTRALER CONFIG")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Check if model is cached
    print(f"\n🗄️ Cache-Check für {config.MODEL_NAME}...")
    check_model_in_cache(config.MODEL_NAME)
    
    # Auto-detect GPU and optimize
    config = detect_gpu_and_optimize(config)
    
    # Check for GGUF availability
    gguf_available = True
    try:
        import subprocess
        subprocess.run([sys.executable, "-c", "import gguf"], check=True, capture_output=True)
    except:
        gguf_available = False
        if config.SAVE_GGUF:
            print("⚠️ GGUF nicht verfügbar - wird deaktiviert")
            config.SAVE_GGUF = False
            config.GGUF_QUANTIZATIONS = []
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Setup LoRA
        model = setup_lora(model, config)
        
        # Prepare dataset
        dataset, tokenizer = prepare_dataset(config, tokenizer)
        
        # Setup trainer
        trainer = setup_trainer(model, tokenizer, dataset, config)
        
        # Train model
        trainer_stats = train_model(trainer, config)
        
        # Test inference
        test_inference(model, tokenizer, config)
        
        # Save model locally
        save_model(model, tokenizer, config)
        
        # Upload to Hub
        upload_to_hub(model, tokenizer, config)
        
        print("\n" + "=" * 60)
        print("🎉 Fine-tuning Pipeline erfolgreich abgeschlossen!")
        print("=" * 60)
        
        # Final summary
        if config.SAVE_LOCAL:
            print(f"📁 Lokale Dateien: ./{config.CUSTOM_MODEL_DIR}/{config.OUTPUT_MODEL_NAME}/")
        if config.UPLOAD_TO_HF:
            print(f"☁️ HF Repository: https://huggingface.co/{config.HF_REPO_ID}")
        
        print("\n💡 Für Evaluierung:")
        print("   python 07_benchmark.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Fehler in Training Pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()