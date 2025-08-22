# -*- coding: utf-8 -*-
"""
GPT-OSS Fine-tuning Script - With central configuration
Fine-tuning script for OpenAI GPT-OSS models using central pipeline_config.json
"""

import os
import sys
import time
import torch
import json
from datetime import datetime
from pathlib import Path

# ========================================
# LOAD CENTRAL CONFIGURATION
# ========================================
sys.path.append(str(Path(__file__).parent.parent.parent))  # To main directory
from config_loader import PipelineConfigLoader

# Load configuration for this module
config_loader = PipelineConfigLoader()
ft_config = config_loader.get_finetuning_config()
tokens = config_loader.get_tokens()

# Show loaded configuration
print("=" * 60)
print("📋 CONFIGURATION LOADED (06_oss_finetuning)")
print("=" * 60)
config_loader.print_config_summary()
print(f"\n  🎯 Model: {ft_config.get('model_name', 'Unknown')}")
print(f"  🤖 Base: {ft_config.get('base_model', 'Unknown')}")
print(f"  📊 Profile: {ft_config.get('training_profile', 'Unknown')}")
print(f"  🔄 Epochs: {ft_config.get('num_train_epochs', 0)}")
print(f"  💾 Merged: {'✅' if ft_config.get('save_merged') else '❌'}")
print(f"  🎯 GGUF: {'✅' if ft_config.get('save_gguf') else '❌'}")
print(f"  🔑 HF-Token: {'✅' if tokens.get('hf_token') else '❌'}")
print("=" * 60)

# Set environment variables for HF tokens
if tokens.get('hf_token'):
    os.environ["HF_TOKEN"] = tokens['hf_token']
    os.environ["HUGGINGFACE_TOKEN"] = tokens['hf_token']
if tokens.get('hf_write_token'):
    os.environ["HF_WRITE_TOKEN"] = tokens['hf_write_token']

# Import unsloth for GPT-OSS
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi

# Configuration Class with values from central config
class Config:
    # Load from central config - with GPT-OSS defaults
    MODEL_NAME = ft_config.get('base_model', 'unsloth/gpt-oss-20b')
    OUTPUT_MODEL_NAME = ft_config.get('model_name', 'GPT-OSS-finetuned')
    HF_REPO_ID = ft_config.get('hf_repo_id', f"user/{ft_config.get('model_name', 'model')}")
    DATASET_PATH = ft_config.get('dataset_path', 'INPUT/thinking_dataset.jsonl')
    CUSTOM_MODEL_DIR = ft_config.get('custom_model_dir', 'CustomModel')
    
    # GPT-OSS specific settings
    REASONING_EFFORT = ft_config.get('reasoning_effort', 'medium')  # low, medium, high
    
    # Training settings
    MAX_SEQ_LENGTH = ft_config.get('max_seq_length', 1024)  # GPT-OSS default
    LOAD_IN_4BIT = ft_config.get('load_in_4bit', True)
    FULL_FINETUNING = ft_config.get('full_finetuning', False)
    
    # LoRA settings - GPT-OSS specific target modules
    LORA_R = ft_config.get('lora_r', 8)
    LORA_ALPHA = ft_config.get('lora_alpha', 16)
    LORA_DROPOUT = ft_config.get('lora_dropout', 0)
    BIAS = ft_config.get('bias', 'none')
    RANDOM_STATE = ft_config.get('random_state', 3407)
    USE_RSLORA = ft_config.get('use_rslora', False)
    USE_GRADIENT_CHECKPOINTING = ft_config.get('use_gradient_checkpointing', 'unsloth')
    
    # GPT-OSS target modules
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    
    # Training parameters
    PER_DEVICE_TRAIN_BATCH_SIZE = ft_config.get('per_device_train_batch_size', 1)
    GRADIENT_ACCUMULATION_STEPS = ft_config.get('gradient_accumulation_steps', 4)
    WARMUP_STEPS = ft_config.get('warmup_steps', 5)
    NUM_TRAIN_EPOCHS = ft_config.get('num_train_epochs', 1)
    MAX_STEPS = ft_config.get('max_steps', 200)
    LEARNING_RATE = ft_config.get('learning_rate', 2e-4)
    LOGGING_STEPS = ft_config.get('logging_steps', 1)
    OPTIM = ft_config.get('optim', 'adamw_8bit')
    WEIGHT_DECAY = ft_config.get('weight_decay', 0.01)
    LR_SCHEDULER_TYPE = ft_config.get('lr_scheduler_type', 'linear')
    SEED = ft_config.get('seed', 3407)
    
    # Output settings
    SAVE_LOCAL = True
    SAVE_LORA = ft_config.get('save_lora', True)
    SAVE_MERGED = ft_config.get('save_merged', False)  # GPT-OSS might not support merging yet
    SAVE_GGUF = ft_config.get('save_gguf', False)  # GPT-OSS GGUF support TBD
    UPLOAD_TO_HF = ft_config.get('upload_to_hf', False)
    
    # Inference settings
    TEMPERATURE = ft_config.get('temperature', 1.0)
    TOP_P = ft_config.get('top_p', 0.95)
    TOP_K = ft_config.get('top_k', 64)
    MAX_NEW_TOKENS = ft_config.get('max_new_tokens', 64)
    
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
                    print(f"✅ Model already in cache: {model_name} ({total_size:.1f} GB)")
                    return True
    
    print(f"📥 Model will be downloaded on first load: {model_name}")
    return False

def detect_gpu_and_optimize(config):
    """Automatically detect GPU and optimize settings for GPT-OSS"""
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
    
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_name = gpu_props.name
    gpu_memory_gb = round(gpu_props.total_memory / 1024 / 1024 / 1024, 1)
    
    config.DETECTED_GPU_NAME = gpu_name
    config.DETECTED_GPU_MEMORY = gpu_memory_gb
    
    print(f"🎯 Detected GPU: {gpu_name} ({gpu_memory_gb} GB VRAM)")
    
    # Enable GPU-specific optimizations
    torch.backends.cudnn.benchmark = True
    
    # TF32 optimizations for newer GPUs
    if any(x in gpu_name.upper() for x in ['A4000', 'A4500', 'A5000', 'A6000', 'RTX 30', 'RTX 40', 'RTX 3', 'RTX 4']):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ Enabled TF32 optimizations (Ampere+ GPU detected)")
    
    # Determine model size (20B or 120B)
    is_120b = "120b" in config.MODEL_NAME.lower()
    model_size = "120B" if is_120b else "20B"
    print(f"📊 Model size: {model_size}")
    
    # Auto-adjust settings based on GPU memory for GPT-OSS
    if is_120b:
        # 120B model requires significant resources
        if gpu_memory_gb < 40:
            print("⚠️ 120B model requires at least 40GB VRAM. Switching to 20B model.")
            config.MODEL_NAME = config.MODEL_NAME.replace("120b", "20b")
            is_120b = False
        else:
            config.PER_DEVICE_TRAIN_BATCH_SIZE = 1
            config.GRADIENT_ACCUMULATION_STEPS = 8
            config.MAX_SEQ_LENGTH = min(config.MAX_SEQ_LENGTH, 512)
    
    if not is_120b:  # 20B model
        if gpu_memory_gb >= 24:
            config.PER_DEVICE_TRAIN_BATCH_SIZE = 2
            config.GRADIENT_ACCUMULATION_STEPS = 2
            config.MAX_SEQ_LENGTH = min(config.MAX_SEQ_LENGTH, 1024)
        elif gpu_memory_gb >= 16:
            config.PER_DEVICE_TRAIN_BATCH_SIZE = 1
            config.GRADIENT_ACCUMULATION_STEPS = 4
            config.MAX_SEQ_LENGTH = min(config.MAX_SEQ_LENGTH, 1024)
        elif gpu_memory_gb >= 12:
            config.PER_DEVICE_TRAIN_BATCH_SIZE = 1
            config.GRADIENT_ACCUMULATION_STEPS = 4
            config.MAX_SEQ_LENGTH = min(config.MAX_SEQ_LENGTH, 512)
        else:
            print("⚠️ GPT-OSS 20B may require at least 12GB VRAM for training")
            config.PER_DEVICE_TRAIN_BATCH_SIZE = 1
            config.GRADIENT_ACCUMULATION_STEPS = 8
            config.MAX_SEQ_LENGTH = min(config.MAX_SEQ_LENGTH, 256)
    
    effective_batch_size = config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS
    print(f"📈 Effective batch size: {effective_batch_size}")
    print(f"📏 Max sequence length: {config.MAX_SEQ_LENGTH}")
    
    return config

def load_model_and_tokenizer(config):
    """Load the GPT-OSS model and tokenizer"""
    print(f"Loading GPT-OSS model: {config.MODEL_NAME}")
    
    if not config.HF_TOKEN:
        print("⚠️ No HF_TOKEN found. Make sure you're logged in with 'huggingface-cli login'")
    
    # List of available GPT-OSS models
    fourbit_models = [
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-20b",
        "unsloth/gpt-oss-120b",
    ]
    
    print(f"Available GPT-OSS models: {', '.join(fourbit_models)}")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.MODEL_NAME,
            dtype=None,  # Auto detection
            max_seq_length=config.MAX_SEQ_LENGTH,
            load_in_4bit=config.LOAD_IN_4BIT,
            token=config.HF_TOKEN,
        )
        
        print("✅ GPT-OSS model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\n🔧 Authentication error. Please check HF_TOKEN in pipeline_config.json")
        raise e

def setup_lora(model, config):
    """Setup LoRA adapters for GPT-OSS"""
    print("Setting up LoRA adapters for GPT-OSS...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        target_modules=config.TARGET_MODULES,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias=config.BIAS,
        use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
        random_state=config.RANDOM_STATE,
        use_rslora=config.USE_RSLORA,
        loftq_config=None,
    )
    
    print("✅ LoRA adapters configured for GPT-OSS!")
    return model

def prepare_dataset(config, tokenizer):
    """Load and prepare dataset for GPT-OSS"""
    print(f"Loading dataset from: {config.DATASET_PATH}")
    
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {config.DATASET_PATH}")
    
    try:
        # Load dataset (JSONL format for GPT-OSS)
        dataset = load_dataset("json", data_files=config.DATASET_PATH, split="train")
        print(f"Dataset size: {len(dataset)} examples")
        
        # Standardize to ShareGPT format (GPT-OSS specific)
        dataset = standardize_sharegpt(dataset)
        
        # Format dataset for training
        def formatting_prompts_func(examples):
            convos = examples["messages"]
            texts = []
            for convo in convos:
                # Apply chat template without generation prompt for training
                text = tokenizer.apply_chat_template(
                    convo, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                texts.append(text)
            return {"text": texts}
        
        dataset = dataset.map(formatting_prompts_func, batched=True)
        
        print("✅ Dataset prepared successfully for GPT-OSS!")
        
        # Show first example
        if len(dataset) > 0:
            print("\n📋 First training example:")
            print(dataset[0]['text'][:500] + "..." if len(dataset[0]['text']) > 500 else dataset[0]['text'])
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error preparing dataset: {e}")
        raise e

def setup_trainer(model, tokenizer, dataset, config):
    """Setup the SFT trainer for GPT-OSS"""
    print("Setting up trainer for GPT-OSS...")
    
    # Training configuration
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=SFTConfig(
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
            output_dir=f"./results/{config.CUSTOM_MODEL_DIR}/{config.OUTPUT_MODEL_NAME}",
            report_to="none",
            
            # GPT-OSS specific settings
            max_seq_length=config.MAX_SEQ_LENGTH,
            packing=False,  # Important for GPT-OSS
            
            # Precision settings
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
        ),
    )
    
    print("✅ Trainer configured for GPT-OSS!")
    return trainer

def train_model(trainer, config):
    """Train the GPT-OSS model"""
    print("🚀 Starting GPT-OSS training...")
    
    start_memory, max_memory = show_memory_stats(config)
    
    # Train
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
    """Test the trained GPT-OSS model with reasoning effort feature"""
    print("\n🧪 Testing inference with different reasoning efforts...")
    
    test_question = "Solve x^5 + 3x^4 - 10 = 3."
    
    # Test with different reasoning efforts
    reasoning_efforts = ["low", "medium", "high"]
    
    for effort in reasoning_efforts:
        print(f"\n--- Testing with reasoning_effort: {effort} ---")
        print(f"User: {test_question}")
        print(f"Assistant ({effort}): ", end="")
        
        messages = [
            {"role": "user", "content": test_question}
        ]
        
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort=effort,  # GPT-OSS specific feature
            ).to(model.device)
            
            _ = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                streamer=TextStreamer(tokenizer, skip_prompt=True),
            )
            
            print()  # New line after response
            
        except Exception as e:
            print(f"❌ Error in inference test with {effort}: {e}")
    
    print("\n✅ Inference tests completed!")

def test_multilingual_inference(model, tokenizer, config):
    """Test multilingual capabilities (from the training dataset)"""
    print("\n🌍 Testing multilingual inference...")
    
    # Test in German (as shown in the original notebook)
    messages = [
        {"role": "system", "content": "reasoning language: German\n\nYou are a helpful assistant that can explain several concepts and problems."},
        {"role": "user", "content": "Erklären Sie das Konzept der künstlichen Intelligenz."}
    ]
    
    print("--- Multilingual Test (German) ---")
    print(f"User: {messages[1]['content']}")
    print("Assistant: ", end="")
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=config.REASONING_EFFORT,
        ).to(model.device)
        
        _ = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            streamer=TextStreamer(tokenizer, skip_prompt=True),
        )
        
        print("\n✅ Multilingual test completed!")
        
    except Exception as e:
        print(f"❌ Error in multilingual test: {e}")

def save_model(model, tokenizer, config):
    """Save the trained GPT-OSS model"""
    if not config.SAVE_LOCAL:
        print("🚫 Local saving skipped")
        return
    
    # Create directory
    custom_model_path = os.path.join(config.CUSTOM_MODEL_DIR, config.OUTPUT_MODEL_NAME)
    os.makedirs(config.CUSTOM_MODEL_DIR, exist_ok=True)
    os.makedirs(custom_model_path, exist_ok=True)
    
    print(f"\n💾 Saving GPT-OSS model: {custom_model_path}")
    
    saved_formats = []
    
    try:
        # Save LoRA adapters
        if config.SAVE_LORA:
            print("📌 Saving LoRA adapters...")
            model.save_pretrained(custom_model_path)
            tokenizer.save_pretrained(custom_model_path)
            saved_formats.append("LoRA Adapter")
            print("✅ LoRA adapters saved!")
            
            # Save config with reasoning effort default
            config_file = os.path.join(custom_model_path, "training_config.json")
            with open(config_file, 'w') as f:
                json.dump({
                    "base_model": config.MODEL_NAME,
                    "reasoning_effort_default": config.REASONING_EFFORT,
                    "max_seq_length": config.MAX_SEQ_LENGTH,
                    "training_epochs": config.NUM_TRAIN_EPOCHS,
                }, f, indent=2)
        
        # Note: GPT-OSS merging and GGUF conversion may not be supported yet
        if config.SAVE_MERGED:
            print("⚠️ Model merging for GPT-OSS is experimental and may not be fully supported")
            # You can try to implement merging here if supported
        
        if config.SAVE_GGUF:
            print("⚠️ GGUF conversion for GPT-OSS models is not yet available")
        
        # Summary
        print(f"\n✅ Local saving completed!")
        print(f"📁 Save location: {config.CUSTOM_MODEL_DIR}/")
        print(f"📋 Saved formats:")
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
        'model_type': 'gpt-oss',
        'reasoning_effort_default': config.REASONING_EFFORT,
        'created_timestamp': datetime.now().isoformat(),
        'saved_formats': saved_formats,
        'config': {
            'max_seq_length': config.MAX_SEQ_LENGTH,
            'load_in_4bit': config.LOAD_IN_4BIT,
            'epochs': config.NUM_TRAIN_EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'reasoning_effort': config.REASONING_EFFORT
        }
    }
    
    # Save info file
    info_file = os.path.join(config.CUSTOM_MODEL_DIR, f"{config.OUTPUT_MODEL_NAME}_info.json")
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Model info saved: {info_file}")
    
    # Create global info file
    global_info_file = "latest_oss_model_info.json"
    with open(global_info_file, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Global model info created: {global_info_file}")

def upload_to_hub(model, tokenizer, config):
    """Upload GPT-OSS model to Hugging Face Hub"""
    if not config.UPLOAD_TO_HF:
        print("🚫 HF upload skipped")
        return
    
    upload_token = config.HF_WRITE_TOKEN or config.HF_TOKEN
    
    if not upload_token:
        print("❌ No HF token available. Upload skipped.")
        return
    
    print(f"\n☁️ Uploading to Hugging Face Hub: {config.HF_REPO_ID}")
    
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
            print("📌 Uploading LoRA adapters...")
            model.push_to_hub(config.HF_REPO_ID, token=upload_token)
            tokenizer.push_to_hub(config.HF_REPO_ID, token=upload_token)
            print("✅ LoRA adapters uploaded!")
        
        print("✅ Upload to Hub successfully completed!")
        
    except Exception as e:
        print(f"❌ Error uploading to Hub: {e}")

def main():
    """Main function"""
    print("🎯 GPT-OSS Fine-tuning - WITH CENTRAL CONFIG")
    print("🤖 OpenAI GPT-OSS Model Training Pipeline")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Show GPT-OSS specific features
    print("\n📋 GPT-OSS Features:")
    print(f"  • Reasoning Effort: {config.REASONING_EFFORT}")
    print(f"  • Model: {config.MODEL_NAME}")
    print(f"  • Dataset: {config.DATASET_PATH}")
    print("=" * 60)
    
    # Check if model is cached
    print(f"\n🗄️ Cache check for {config.MODEL_NAME}...")
    check_model_in_cache(config.MODEL_NAME)
    
    # Auto-detect GPU and optimize
    config = detect_gpu_and_optimize(config)
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config)
        
        # Setup LoRA
        model = setup_lora(model, config)
        
        # Prepare dataset
        dataset = prepare_dataset(config, tokenizer)
        
        # Setup trainer
        trainer = setup_trainer(model, tokenizer, dataset, config)
        
        # Train model
        trainer_stats = train_model(trainer, config)
        
        # Test inference with reasoning efforts
        test_inference(model, tokenizer, config)
        
        # Test multilingual capabilities
        test_multilingual_inference(model, tokenizer, config)
        
        # Save model locally
        save_model(model, tokenizer, config)
        
        # Upload to Hub
        upload_to_hub(model, tokenizer, config)
        
        print("\n" + "=" * 60)
        print("🎉 GPT-OSS Fine-tuning pipeline successfully completed!")
        print("=" * 60)
        
        # Final summary
        if config.SAVE_LOCAL:
            print(f"📁 Local files: ./{config.CUSTOM_MODEL_DIR}/{config.OUTPUT_MODEL_NAME}/")
        if config.UPLOAD_TO_HF:
            print(f"☁️ HF Repository: https://huggingface.co/{config.HF_REPO_ID}")
        
        print("\n💡 Usage tips:")
        print("   • Use reasoning_effort='low' for fast responses")
        print("   • Use reasoning_effort='medium' for balanced performance")  
        print("   • Use reasoning_effort='high' for complex reasoning tasks")
        print("\n💡 For evaluation:")
        print("   python 07_oss_benchmark.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()