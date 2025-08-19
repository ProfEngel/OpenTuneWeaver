#!/usr/bin/env python3

"""
OpenTuneWeaver UI - Universal Path Handling with Expert Mode
Version: 11.0 - Enhanced with Expert Settings and Help Page
Location: Place this file in OpenTuneWeaver/ root directory
"""

import gradio as gr
import os
import sys
import shutil
import subprocess
import time
import json
import zipfile
import getpass
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import threading
import tempfile
import signal

# ==================== PATH CONFIGURATION ====================

def get_project_root():
    """
    Gets the OpenTuneWeaver project root directory.
    This app should be placed directly in the OpenTuneWeaver root.
    """
    # Current file location (should be OpenTuneWeaver/app.py)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    
    # Verify we're in the right location
    if not (project_root / "pipeline").exists():
        print(f"⚠️ Warning: pipeline directory not found at {project_root / 'pipeline'}")
        print(f"   Current location: {current_file}")
        print(f"   Make sure app.py is in the OpenTuneWeaver root directory!")
    
    return project_root

# Global paths
PROJECT_ROOT = get_project_root()
PIPELINE_PATH = PROJECT_ROOT / "pipeline"
UI_PATH = PROJECT_ROOT / "ui"
WORKSPACE_PATH = PROJECT_ROOT.parent  # /workspace on RunPod, parent dir locally

# Detect environment
IS_RUNPOD = "/workspace" in str(PROJECT_ROOT)

print("="*60)
print("🎯 OpenTuneWeaver Path Configuration")
print("="*60)
print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"📁 Pipeline Path: {PIPELINE_PATH}")
print(f"📁 UI Path: {UI_PATH}")
print(f"🌐 Environment: {'RunPod' if IS_RUNPOD else 'Local'}")
print(f"✅ Pipeline exists: {PIPELINE_PATH.exists()}")
print("="*60)

# Verify critical paths
if not PIPELINE_PATH.exists():
    print("❌ CRITICAL: Pipeline directory not found!")
    print("   Creating pipeline structure...")
    PIPELINE_PATH.mkdir(parents=True, exist_ok=True)

def get_pipeline_file(relative_path):
    """Helper function to get correct pipeline file path."""
    return PIPELINE_PATH / relative_path

def get_ui_file(relative_path):
    """Helper function to get correct UI file path."""
    return UI_PATH / relative_path

# ==================== CONSTANTS ====================

AVAILABLE_MODELS = [
    "unsloth/gemma-3-270m-it",
    "unsloth/gemma-3-1b-it",
    "unsloth/gemma-3-4b-it",
    "unsloth/gemma-3-12b-it",
    "unsloth/gemma-3-27b-it",
    "unsloth/gemma-3n-E2B-it",
    "unsloth/gemma-3n-E4B-it"
]

FINETUNING_PRESETS = {
    "Test": {
        "description": "Quick test run (minimal resources)",
        "max_seq_length": 2048,
        "num_train_epochs": 1,
        "learning_rate": 5e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 50,
        "lora_r": 4,
        "lora_alpha": 4
    },
    "Development": {
        "description": "Development testing (balanced)",
        "max_seq_length": 4096,
        "num_train_epochs": 2,
        "learning_rate": 3e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 100,
        "lora_r": 8,
        "lora_alpha": 8
    },
    "Production": {
        "description": "Production quality (best results)",
        "max_seq_length": 8192,
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "warmup_steps": 200,
        "lora_r": 16,
        "lora_alpha": 16
    },
    "Expert": {
        "description": "Custom settings (use expert page)",
        "max_seq_length": 8192,
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "warmup_steps": 200,
        "lora_r": 8,
        "lora_alpha": 8
    }
}

# Global variables
progress_messages = []
processing_active = False
current_process = None
current_preset = "Production"
pipeline_status = {}
selected_steps = [1, 2, 3, 4, 5, 6, 7, 8]  # Default: all steps

# ==================== LOGGING ====================

def log_message(message: str, level: str = "INFO"):
    """Adds a log message."""
    global progress_messages
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {level}: {message}"
    progress_messages.append(formatted_message)
    print(formatted_message)
    return formatted_message

def get_terminal_output():
    """Returns all log messages."""
    global progress_messages
    return '\n'.join(progress_messages[-100:])  # Last 100 messages

def clear_terminal():
    """Clears terminal output."""
    global progress_messages
    progress_messages = []
    return ""

# ==================== FILE UPLOAD ====================

def save_uploaded_files(files: List) -> str:
    """Saves uploaded files to the upload directory."""
    if not files:
        return "❌ No files selected"

    upload_dir = get_pipeline_file("modules/01_convert/UPLOAD")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    log_message(f"📁 Upload directory: {upload_dir}")

    # Delete old files
    for old_file in upload_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink()
            log_message(f"🗑️ Deleted old file: {old_file.name}")

    saved_count = 0
    for file in files:
        if file is not None:
            try:
                file_path = Path(file.name)
                target_path = upload_dir / file_path.name
                shutil.copy2(file_path, target_path)
                log_message(f"✅ File saved: {file_path.name}")
                saved_count += 1
            except Exception as e:
                log_message(f"❌ Error saving file: {e}", "ERROR")

    return f"✅ {saved_count} files successfully uploaded to {upload_dir.relative_to(PROJECT_ROOT)}"

# ==================== PIPELINE STATUS ====================

def get_pipeline_status_file():
    """Returns the path to the pipeline status file."""
    return get_pipeline_file("pipeline_status.json")

def reset_pipeline_status():
    """Resets the pipeline status."""
    global pipeline_status
    pipeline_status = {
        1: {"name": "Document Conversion", "status": "pending", "icon": "📄", "stats": {}, "duration": None},
        2: {"name": "Wiki Generation", "status": "pending", "icon": "📚", "stats": {}, "duration": None},
        3: {"name": "QA Creation", "status": "pending", "icon": "❓", "stats": {}, "duration": None},
        4: {"name": "Dataset Formatting", "status": "pending", "icon": "🔧", "stats": {}, "duration": None},
        5: {"name": "Benchmark Creation", "status": "pending", "icon": "📊", "stats": {}, "duration": None},
        6: {"name": "Fine-tuning", "status": "pending", "icon": "🤖", "stats": {}, "duration": None},
        7: {"name": "Benchmarking", "status": "pending", "icon": "🏆", "stats": {}, "duration": None},
        8: {"name": "Results Archive", "status": "pending", "icon": "📦", "stats": {}, "duration": None}
    }
    
    # Write initial status to file
    status_file = get_pipeline_status_file()
    status_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to string keys for JSON
    json_status = {str(k): v for k, v in pipeline_status.items()}
    
    with open(status_file, 'w') as f:
        json.dump(json_status, f)

def read_pipeline_status():
    """Reads the pipeline status from file."""
    global pipeline_status
    status_file = get_pipeline_status_file()
    
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                file_status = json.load(f)
                # Update global status - convert string keys to int
                for step_id, step_data in file_status.items():
                    try:
                        int_id = int(step_id)
                        if int_id in range(1, 9):
                            pipeline_status[int_id] = step_data
                    except ValueError:
                        pass
        except:
            pass
    
    return pipeline_status

def create_pipeline_overview():
    """Creates HTML for pipeline status overview."""
    global pipeline_status
    
    # Read latest status from file
    read_pipeline_status()
    
    # Ensure all steps exist in status
    if not pipeline_status:
        reset_pipeline_status()
    
    html = """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 12px; color: white;">
        <h3 style="text-align: center; margin: 0 0 15px 0; font-size: 1.2em;">🚀 Pipeline Progress</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px;">
    """
    
    for step_id in range(1, 9):
        step_info = pipeline_status.get(step_id, {
            "name": f"Step {step_id}",
            "status": "pending",
            "icon": "📋",
            "stats": {},
            "duration": None
        })
        
        status = step_info.get("status", "pending")
        icon = step_info.get("icon", "")
        name = step_info.get("name", "")
        stats = step_info.get("stats", {})
        duration = step_info.get("duration", None)
        
        # Status colors and symbols
        if status == "completed":
            bg_color = "#10b981"
            status_icon = "✅"
            border_style = "2px solid #059669"
        elif status == "running":
            bg_color = "#f59e0b"
            status_icon = "⚡"
            border_style = "2px solid #d97706"
        elif status == "failed":
            bg_color = "#ef4444"
            status_icon = "❌"
            border_style = "2px solid #dc2626"
        else:  # pending
            bg_color = "#6b7280"
            status_icon = "⏳"
            border_style = "2px solid #4b5563"
        
        # Build stats text
        stats_text = ""
        if stats:
            if "files_processed" in stats:
                stats_text += f"📁 {stats['files_processed']} files<br>"
            if "entries" in stats:
                stats_text += f"📝 {stats['entries']} entries<br>"
        
        if duration:
            if duration < 60:
                duration_text = f"⏱️ {duration:.0f}s"
            else:
                duration_text = f"⏱️ {duration/60:.1f}m"
            stats_text += duration_text
        
        # Create card for each step
        card_html = f"""
        <div style="background: white; border-radius: 8px; padding: 10px; text-align: center; 
                    border: {border_style}; position: relative; min-height: 120px;">
            <div style="font-size: 1.8em; margin-bottom: 3px;">{icon}</div>
            <div style="color: #1f2937; font-weight: bold; font-size: 0.85em;">Step {step_id}</div>
            <div style="color: #4b5563; font-size: 0.75em; margin: 3px 0;">{name}</div>
            <div style="background: {bg_color}; color: white; padding: 2px 6px; border-radius: 4px; 
                        font-size: 0.7em; margin: 5px 0;">
                {status_icon} {status.upper()}
            </div>
            <div style="color: #6b7280; font-size: 0.65em; margin-top: 3px;">
                {stats_text}
            </div>
        </div>
        """
        
        html += card_html
    
    html += """
        </div>
        <div style="text-align: center; margin-top: 15px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 8px;">
    """
    
    # Add summary
    completed = sum(1 for s in pipeline_status.values() if s.get("status") == "completed")
    running = sum(1 for s in pipeline_status.values() if s.get("status") == "running")
    total = 8
    progress_percent = (completed / total) * 100 if total > 0 else 0
    
    html += f"""
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div style="font-size: 0.95em; min-width: 120px;">
                    Progress: {completed}/{total} Steps
                </div>
                <div style="flex: 1; margin: 0 15px; background: rgba(255,255,255,0.2); border-radius: 8px; height: 8px; min-width: 100px;">
                    <div style="background: #10b981; height: 100%; border-radius: 8px; width: {progress_percent}%; transition: width 0.5s;"></div>
                </div>
                <div style="font-size: 0.95em;">
                    {progress_percent:.0f}%
                </div>
            </div>
    """
    
    if running > 0:
        running_steps = [k for k,v in pipeline_status.items() if v.get('status')=='running']
        if running_steps:
            html += f"""
            <div style="margin-top: 5px; font-size: 0.85em; opacity: 0.9;">
                ⚡ Currently running: Step {running_steps[0]}
            </div>
            """
    
    html += """
        </div>
    </div>
    """
    
    return html

# ==================== STEP SELECTION ====================

def update_step_selection(steps):
    """Updates the selected pipeline steps."""
    global selected_steps
    
    if not steps:
        selected_steps = []
        return "⚠️ Please select at least one step"
    
    # Extract step numbers from labels
    selected_steps = []
    for step_label in steps:
        # Extract number from "1. Document Conversion" format
        step_num = int(step_label.split('.')[0])
        selected_steps.append(step_num)
    
    selected_steps = sorted(selected_steps)
    
    if len(selected_steps) == 8:
        return "✅ All steps selected (1-8)"
    else:
        selected_text = f"✅ Selected steps: {', '.join([str(s) for s in selected_steps])}"
        return selected_text

def get_preset_info(preset_name: str):
    """Returns information about a preset."""
    if preset_name not in FINETUNING_PRESETS:
        return "Select a preset to see details"
    
    preset = FINETUNING_PRESETS[preset_name]
    info = f"**{preset_name} Settings:**\n"
    info += f"- {preset['description']}\n"
    info += f"- Sequence Length: {preset['max_seq_length']}\n"
    info += f"- Training Epochs: {preset['num_train_epochs']}\n"
    info += f"- Learning Rate: {preset['learning_rate']}\n"
    info += f"- LoRA Rank: {preset['lora_r']}\n"
    
    return info

# ==================== CONFIGURATION ====================

def load_existing_config():
    """Loads existing configuration if available."""
    config_file = get_pipeline_file("pipeline_config.json")
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # Ensure all required keys exist
            if "benchmark" in config:
                if "pre_model" not in config["benchmark"]:
                    config["benchmark"]["pre_model"] = {
                        "name": config.get("finetuning", {}).get("base_model", "unsloth/gemma-3n-E2B-it"),
                        "type": "transformers",
                        "load_in_4bit": False,
                        "max_seq_length": 2048
                    }
                if "post_model" not in config["benchmark"]:
                    model_name = config.get("finetuning", {}).get("model_name", "OpenTuneWeaver-Model")
                    custom_dir = config.get("finetuning", {}).get("custom_model_dir", "CustomModel")
                    config["benchmark"]["post_model"] = {
                        "name": f"{custom_dir}/{model_name}",
                        "type": "unknown",
                        "load_in_4bit": False,
                        "max_seq_length": 2048,
                        "base_model": None
                    }
                if "questions_file" not in config["benchmark"]:
                    config["benchmark"]["questions_file"] = "BENCHMARKFRAGEN/benchmark_fragen_complete.json"
            return config
        except Exception as e:
            log_message(f"❌ Error loading configuration: {e}", "ERROR")
    
    # Default configuration with complete structure
    return {
        "tokens": {"hf_token": "", "hf_write_token": ""},
        "api_configs": {
            "01_convert": {
                "use_openai_api": True,
                "openai_base_url": "http://localhost:11434/v1",
                "openai_api_key": "ollama",
                "openai_model_name": "gemma3:12b-it-qat",
                "temperature": 0.1
            },
            "02_genwiki": {
                "use_openai_api": True,
                "openai_base_url": "http://localhost:11434/v1",
                "openai_api_key": "ollama",
                "openai_model_name": "gemma3:12b-it-qat",
                "temperature": 0.3
            },
            "03_instructQA": {
                "use_openai_api": True,
                "openai_base_url": "http://localhost:11434/v1",
                "openai_api_key": "ollama",
                "openai_model_name": "gemma3:12b-it-qat",
                "temperature": 0.7
            },
            "05_bmcreator": {
                "use_openai_api": True,
                "openai_base_url": "http://localhost:11434/v1",
                "openai_api_key": "ollama",
                "openai_model_name": "gemma3:12b-it-qat",
                "temperature": 0.5
            }
        },
        "finetuning": {
            "model_name": "OpenTuneWeaver-Model",
            "base_model": "unsloth/gemma-3n-E2B-it",
            "hf_repo_id": "user/OpenTuneWeaver-Model",
            "dataset_path": "INPUT/dataset.json",
            "chat_template": "gemma-3",
            "custom_model_dir": "CustomModel",
            "max_seq_length": 8192,
            "load_in_4bit": True,
            "full_finetuning": False,
            "lora_r": 8,
            "lora_alpha": 8,
            "lora_dropout": 0,
            "bias": "none",
            "random_state": 3407,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "warmup_steps": 200,
            "num_train_epochs": 3,
            "max_steps": -1,
            "learning_rate": 5e-5,
            "logging_steps": 5,
            "optim": "adamw_8bit",
            "weight_decay": 0.03,
            "lr_scheduler_type": "cosine",
            "seed": 3407,
            "save_lora": True,
            "save_merged": True,
            "save_gguf": False,
            "upload_to_hf": False,
            "gguf_quantizations": ["q8_0"],
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_new_tokens": 128
        },
        "benchmark": {
            "mode": "comparison",
            "pre_model": {
                "name": "unsloth/gemma-3n-E2B-it",
                "type": "transformers",
                "load_in_4bit": False,
                "max_seq_length": 2048
            },
            "post_model": {
                "name": "CustomModel/OpenTuneWeaver-Model",
                "type": "unknown",
                "load_in_4bit": False,
                "max_seq_length": 2048,
                "base_model": None
            },
            "evaluator": {
                "type": "api",
                "api_base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "gemma3:12b-it-qat"
            },
            "questions_file": "BENCHMARKFRAGEN/benchmark_fragen_complete.json",
            "max_new_tokens": 256,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1
        },
        "pipeline": {
            "auto_cleanup": False,
            "verbose": True,
            "continue_on_error": True,
            "save_metrics": True
        }
    }

def save_config_from_quick_settings(
    model_name, hf_token, preset_name, save_lora, save_merged, save_gguf
):
    """Saves configuration from quick settings."""
    global current_preset
    current_preset = preset_name
    
    try:
        # Load existing config as base
        config = load_existing_config()
        
        # Update with quick settings
        config["tokens"]["hf_token"] = hf_token
        config["tokens"]["hf_write_token"] = hf_token
        config["finetuning"]["model_name"] = model_name
        config["finetuning"]["save_lora"] = save_lora
        config["finetuning"]["save_merged"] = save_merged
        config["finetuning"]["save_gguf"] = save_gguf
        
        # Apply preset settings if not Expert
        if preset_name in FINETUNING_PRESETS and preset_name != "Expert":
            preset = FINETUNING_PRESETS[preset_name]
            config["finetuning"]["max_seq_length"] = preset["max_seq_length"]
            config["finetuning"]["num_train_epochs"] = preset["num_train_epochs"]
            config["finetuning"]["learning_rate"] = preset["learning_rate"]
            config["finetuning"]["per_device_train_batch_size"] = preset["batch_size"]
            config["finetuning"]["gradient_accumulation_steps"] = preset["gradient_accumulation_steps"]
            config["finetuning"]["warmup_steps"] = preset["warmup_steps"]
            config["finetuning"]["lora_r"] = preset["lora_r"]
            config["finetuning"]["lora_alpha"] = preset["lora_alpha"]
        
        # Ensure benchmark section has all required fields
        if "benchmark" not in config:
            config["benchmark"] = {}
        
        # Update pre_model and post_model based on finetuning settings
        base_model = config["finetuning"].get("base_model", "unsloth/gemma-3n-E2B-it")
        custom_dir = config["finetuning"].get("custom_model_dir", "CustomModel")
        
        config["benchmark"]["pre_model"] = {
            "name": base_model,
            "type": "transformers",
            "load_in_4bit": False,
            "max_seq_length": 2048
        }
        
        config["benchmark"]["post_model"] = {
            "name": f"{custom_dir}/{model_name}",
            "type": "unknown",
            "load_in_4bit": False,
            "max_seq_length": 2048,
            "base_model": None
        }
        
        # Ensure other benchmark fields exist
        if "mode" not in config["benchmark"]:
            config["benchmark"]["mode"] = "comparison"
        if "evaluator" not in config["benchmark"]:
            config["benchmark"]["evaluator"] = {
                "type": "api",
                "api_base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": "gemma3:12b-it-qat"
            }
        if "questions_file" not in config["benchmark"]:
            config["benchmark"]["questions_file"] = "BENCHMARKFRAGEN/benchmark_fragen_complete.json"
        if "max_new_tokens" not in config["benchmark"]:
            config["benchmark"]["max_new_tokens"] = 256
        if "temperature" not in config["benchmark"]:
            config["benchmark"]["temperature"] = 0.3
        if "top_p" not in config["benchmark"]:
            config["benchmark"]["top_p"] = 0.9
        if "top_k" not in config["benchmark"]:
            config["benchmark"]["top_k"] = 50
        if "repetition_penalty" not in config["benchmark"]:
            config["benchmark"]["repetition_penalty"] = 1.1
        
        # Save config
        config_file = get_pipeline_file("pipeline_config.json")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        log_message(f"✅ Quick settings saved with preset: {preset_name}")
        return f"✅ Settings saved! Using preset: {preset_name}"
        
    except Exception as e:
        error_msg = f"❌ Error saving settings: {e}"
        log_message(error_msg, "ERROR")
        return error_msg

def save_config_from_ui(
    # Tokens
    hf_token, hf_write_token,
    # API Configs
    api_base_url, api_key,
    convert_model, wiki_model, qa_model, benchmark_model,
    convert_temp, wiki_temp, qa_temp, benchmark_temp,
    # Fine-tuning
    model_name, base_model, hf_repo_id, custom_model_dir,
    max_seq_length, load_in_4bit, full_finetuning,
    lora_r, lora_alpha, lora_dropout,
    batch_size, grad_accumulation, warmup_steps, num_epochs,
    learning_rate, weight_decay,
    save_lora, save_merged, save_gguf,
    # Benchmark
    benchmark_mode, evaluator_model, max_new_tokens,
    eval_temp, top_p, top_k, repetition_penalty,
    # Pipeline
    auto_cleanup, verbose_mode, continue_on_error
):
    """Saves the configuration from the expert UI."""
    try:
        config = {
            "version": "11.0",
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "tokens": {
                "hf_token": hf_token,
                "hf_write_token": hf_write_token or hf_token
            },
            "api_configs": {
                "01_convert": {
                    "use_openai_api": True,
                    "openai_base_url": api_base_url,
                    "openai_api_key": api_key,
                    "openai_model_name": convert_model,
                    "temperature": convert_temp
                },
                "02_genwiki": {
                    "use_openai_api": True,
                    "openai_base_url": api_base_url,
                    "openai_api_key": api_key,
                    "openai_model_name": wiki_model,
                    "temperature": wiki_temp
                },
                "03_instructQA": {
                    "use_openai_api": True,
                    "openai_base_url": api_base_url,
                    "openai_api_key": api_key,
                    "openai_model_name": qa_model,
                    "temperature": qa_temp
                },
                "05_bmcreator": {
                    "use_openai_api": True,
                    "openai_base_url": api_base_url,
                    "openai_api_key": api_key,
                    "openai_model_name": benchmark_model,
                    "temperature": benchmark_temp
                }
            },
            "finetuning": {
                "model_name": model_name,
                "base_model": base_model,
                "hf_repo_id": hf_repo_id,
                "dataset_path": "INPUT/dataset.json",
                "chat_template": "gemma-3",
                "custom_model_dir": custom_model_dir,
                "max_seq_length": int(max_seq_length),
                "load_in_4bit": load_in_4bit,
                "full_finetuning": full_finetuning,
                "lora_r": int(lora_r),
                "lora_alpha": int(lora_alpha),
                "lora_dropout": lora_dropout,
                "bias": "none",
                "random_state": 3407,
                "per_device_train_batch_size": int(batch_size),
                "gradient_accumulation_steps": int(grad_accumulation),
                "warmup_steps": int(warmup_steps),
                "num_train_epochs": int(num_epochs),
                "max_steps": -1,
                "learning_rate": learning_rate,
                "logging_steps": 5,
                "optim": "adamw_8bit",
                "weight_decay": weight_decay,
                "lr_scheduler_type": "cosine",
                "seed": 3407,
                "save_lora": save_lora,
                "save_merged": save_merged,
                "save_gguf": save_gguf,
                "upload_to_hf": False,
                "gguf_quantizations": ["q8_0"],
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "max_new_tokens": 128
            },
            "benchmark": {
                "mode": benchmark_mode,
                "pre_model": {
                    "name": base_model,
                    "type": "transformers",
                    "load_in_4bit": False,
                    "max_seq_length": 2048
                },
                "post_model": {
                    "name": f"{custom_model_dir}/{model_name}",
                    "type": "unknown",
                    "load_in_4bit": False,
                    "max_seq_length": 2048,
                    "base_model": None
                },
                "evaluator": {
                    "type": "api",
                    "api_base_url": api_base_url,
                    "api_key": api_key,
                    "model": evaluator_model
                },
                "questions_file": "BENCHMARKFRAGEN/benchmark_fragen_complete.json",
                "max_new_tokens": int(max_new_tokens),
                "temperature": eval_temp,
                "top_p": top_p,
                "top_k": int(top_k),
                "repetition_penalty": repetition_penalty
            },
            "pipeline": {
                "auto_cleanup": auto_cleanup,
                "verbose": verbose_mode,
                "continue_on_error": continue_on_error,
                "save_metrics": True
            }
        }

        config_file = get_pipeline_file("pipeline_config.json")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        log_message("✅ Expert configuration saved")
        return "✅ Expert configuration successfully saved!"
        
    except Exception as e:
        error_msg = f"❌ Error saving configuration: {e}"
        log_message(error_msg, "ERROR")
        return error_msg

# ==================== PIPELINE CONTROL ====================

def get_pipeline_mode():
    """Determines pipeline mode based on selected steps."""
    global selected_steps
    
    if not selected_steps:
        return "none"
    
    steps = sorted(selected_steps)
    
    # Check for standard modes
    if steps == [1, 2, 3, 4, 5, 6, 7, 8]:
        return "full"
    elif steps == [1, 2, 3, 4, 5]:
        return "data"
    elif steps == [6, 7]:
        return "training"
    elif steps == [8]:
        return "archive"
    elif len(steps) == 1:
        return "single"
    else:
        return "custom"

def start_pipeline():
    """Starts the pipeline with selected steps."""
    global processing_active, current_process, selected_steps

    if processing_active:
        return "⚠️ Pipeline already running!", create_pipeline_overview()

    # Reset pipeline status
    reset_pipeline_status()
    
    mode = get_pipeline_mode()
    
    if mode == "none":
        return "❌ No steps selected!", create_pipeline_overview()

    def run_pipeline():
        global processing_active, current_process
        processing_active = True
        log_message(f"🚀 Pipeline started with steps: {selected_steps}")

        try:
            # Use the pipeline directory
            pipeline_dir = PIPELINE_PATH
            
            if not pipeline_dir.exists():
                log_message(f"❌ Pipeline directory not found at: {pipeline_dir}", "ERROR")
                processing_active = False
                return
                
            if not (pipeline_dir / "run_pipeline.py").exists():
                log_message(f"❌ run_pipeline.py not found!", "ERROR")
                processing_active = False
                return

            original_dir = os.getcwd()
            os.chdir(pipeline_dir)
            log_message(f"📁 Changed to pipeline directory: {pipeline_dir}")

            # Create command line arguments
            cmd_args = [
                sys.executable, 
                "-u",  # Unbuffered output
                "run_pipeline.py",
                "--auto",  # Automated mode
                "--mode", mode,
                "--use-existing-config"
            ]
            
            # Add step parameters for custom mode
            if mode == "custom":
                cmd_args.extend(["--start", str(min(selected_steps))])
                cmd_args.extend(["--end", str(max(selected_steps))])
            elif mode == "single":
                cmd_args.extend(["--step", str(selected_steps[0])])
            
            log_message(f"🖥️ Command: {' '.join(cmd_args)}")

            # Start run_pipeline.py with parameters
            current_process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Read output and display in terminal
            for line in iter(current_process.stdout.readline, ''):
                if not line:
                    break
                
                # Log all outputs
                clean_line = line.strip()
                if clean_line:
                    if "🔴" in clean_line or "ERROR" in clean_line or "❌" in clean_line:
                        log_message(clean_line, "ERROR")
                    elif "⚠️" in clean_line or "WARNING" in clean_line:
                        log_message(clean_line, "WARNING")
                    elif "✅" in clean_line or "🎉" in clean_line:
                        log_message(clean_line, "SUCCESS")
                    else:
                        log_message(clean_line, "INFO")

            # Wait for pipeline to finish
            return_code = current_process.wait()
            
            os.chdir(original_dir)

            if return_code == 0:
                log_message("🎉 Pipeline completed successfully!")
            else:
                log_message(f"❌ Pipeline failed (Code: {return_code})", "ERROR")

        except Exception as e:
            log_message(f"❌ Pipeline error: {e}", "ERROR")
        finally:
            processing_active = False
            current_process = None
            if 'original_dir' in locals():
                os.chdir(original_dir)

    # Start pipeline in separate thread
    pipeline_thread = threading.Thread(target=run_pipeline)
    pipeline_thread.daemon = True
    pipeline_thread.start()

    step_list = ', '.join(str(s) for s in sorted(selected_steps))
    return f"🚀 Pipeline started - Steps: {step_list}", create_pipeline_overview()

def stop_pipeline():
    """Stops the running pipeline."""
    global processing_active, current_process

    if not processing_active or current_process is None:
        return "⚠️ No active pipeline found", create_pipeline_overview()

    try:
        # Try graceful termination
        current_process.terminate()
        time.sleep(2)
        
        # If still active, force kill
        if current_process.poll() is None:
            current_process.kill()
        
        log_message("⏹️ Pipeline stopped", "WARNING")
        processing_active = False
        current_process = None
        
        return "⏹️ Pipeline stopped", create_pipeline_overview()
    except Exception as e:
        log_message(f"❌ Error stopping pipeline: {e}", "ERROR")
        return f"❌ Error stopping pipeline: {e}", create_pipeline_overview()

# ==================== DOWNLOAD FUNCTIONS ====================

def create_documents_zip():
    """Creates a ZIP file with all documents (without models)."""
    try:
        log_message("📦 Creating documents ZIP...")

        temp_dir = tempfile.mkdtemp()
        zip_path = Path(temp_dir) / f"opentuneweaver_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            total_files = 0
            modules_dir = get_pipeline_file("modules")
            
            # Search in all module directories
            if modules_dir.exists():
                document_dirs = [
                    ("01_convert", ["OUTPUT", "INPUT", "UPLOAD"]),
                    ("02_wiki", ["OUTPUT", "INPUT"]), 
                    ("03_instructQA", ["OUTPUT", "INPUT"]),
                    ("04_format", ["OUTPUT", "INPUT"]),
                    ("05_bmcreator", ["OUTPUT", "INPUT", "BENCHMARKFRAGEN"]),
                    ("07_benchmark", ["OUTPUT", "INPUT", "BENCHMARKFRAGEN"])
                ]

                for module_name, folders in document_dirs:
                    module_dir = modules_dir / module_name
                    if module_dir.exists():
                        for folder in folders:
                            folder_path = module_dir / folder
                            if folder_path.exists():
                                for file_path in folder_path.rglob("*"):
                                    if file_path.is_file() and not file_path.suffix.lower() in ['.bin', '.safetensors', '.pt', '.pth', '.gguf']:
                                        rel_path = file_path.relative_to(folder_path)
                                        arcname = f"{module_name}_{folder}/{rel_path}"
                                        zipf.write(file_path, arcname)
                                        total_files += 1

            if total_files == 0:
                return None, "❌ No documents found for download"

            log_message(f"✅ Documents ZIP created ({total_files} files)")
            return str(zip_path), f"✅ Documents ZIP created ({total_files} files)"

    except Exception as e:
        error_msg = f"❌ Error creating documents ZIP: {e}"
        log_message(error_msg, "ERROR")
        return None, error_msg

def create_model_zip():
    """Creates a ZIP file with all model files."""
    try:
        log_message("📦 Creating model ZIP...")

        temp_dir = tempfile.mkdtemp()
        zip_path = Path(temp_dir) / f"opentuneweaver_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            total_files = 0
            
            # CustomModel directory
            custom_model_paths = [
                get_pipeline_file("modules/06_finetuning/CustomModel"),
                get_pipeline_file("CustomModel")
            ]
            
            for custom_model_dir in custom_model_paths:
                if custom_model_dir.exists():
                    for file_path in custom_model_dir.rglob("*"):
                        if file_path.is_file():
                            rel_path = file_path.relative_to(custom_model_dir)
                            arcname = f"custom_model/{rel_path}"
                            zipf.write(file_path, arcname)
                            total_files += 1

            if total_files == 0:
                return None, "❌ No model files found for download"

            file_size_mb = zip_path.stat().st_size / (1024 * 1024)
            log_message(f"✅ Model ZIP created ({total_files} files, {file_size_mb:.1f} MB)")
            return str(zip_path), f"✅ Model ZIP created ({total_files} files, {file_size_mb:.1f} MB)"

    except Exception as e:
        error_msg = f"❌ Error creating model ZIP: {e}"
        log_message(error_msg, "ERROR")
        return None, error_msg

# ==================== CLEANUP ====================

def get_cleanup_info():
    """Returns information about folders to be deleted."""
    try:
        modules_dir = get_pipeline_file("modules")
        
        if not modules_dir.exists():
            return "❌ Pipeline directory not found"

        cleanup_info = []
        total_files = 0
        total_size = 0

        module_folders = [
            ("01_convert", ["INPUT", "OUTPUT", "UPLOAD"]),
            ("02_wiki", ["INPUT", "OUTPUT"]),
            ("03_instructQA", ["INPUT", "OUTPUT"]),
            ("04_format", ["INPUT", "OUTPUT"]),
            ("05_bmcreator", ["INPUT", "OUTPUT", "BENCHMARKFRAGEN"]),
            ("06_finetuning", ["INPUT", "OUTPUT"]),
            ("07_benchmark", ["INPUT", "OUTPUT", "BENCHMARKFRAGEN"])
        ]

        for module_name, folders in module_folders:
            module_dir = modules_dir / module_name
            if module_dir.exists():
                for folder in folders:
                    folder_path = module_dir / folder
                    if folder_path.exists():
                        file_count = 0
                        folder_size = 0
                        for file_path in folder_path.rglob("*"):
                            if file_path.is_file():
                                file_count += 1
                                folder_size += file_path.stat().st_size

                        if file_count > 0:
                            size_mb = folder_size / (1024 * 1024)
                            cleanup_info.append(f"📁 {module_name}/{folder}: {file_count} files ({size_mb:.1f} MB)")
                            total_files += file_count
                            total_size += folder_size

        if not cleanup_info:
            return "✅ No files found for cleanup"

        total_size_mb = total_size / (1024 * 1024)
        info_text = f"🗑️ Cleanup Overview:\n\n"
        info_text += "\n".join(cleanup_info)
        info_text += f"\n\n📊 Total: {total_files} files ({total_size_mb:.1f} MB)"
        info_text += f"\n\n⚠️ WARNING: This action cannot be undone!"

        return info_text

    except Exception as e:
        return f"❌ Error analyzing files: {e}"

def cleanup_pipeline_folders():
    """Deletes all working directories."""
    try:
        log_message("🗑️ Starting cleanup...")
        modules_dir = get_pipeline_file("modules")

        if not modules_dir.exists():
            return "❌ Pipeline directory not found"

        deleted_files = 0
        deleted_folders = 0

        module_folders = [
            ("01_convert", ["INPUT", "OUTPUT", "UPLOAD"]),
            ("02_wiki", ["INPUT", "OUTPUT"]),
            ("03_instructQA", ["INPUT", "OUTPUT"]),
            ("04_format", ["INPUT", "OUTPUT"]),
            ("05_bmcreator", ["INPUT", "OUTPUT", "BENCHMARKFRAGEN"]),
            ("06_finetuning", ["INPUT", "OUTPUT"]),
            ("07_benchmark", ["INPUT", "OUTPUT", "BENCHMARKFRAGEN"])
        ]

        for module_name, folders in module_folders:
            module_dir = modules_dir / module_name
            if module_dir.exists():
                for folder in folders:
                    folder_path = module_dir / folder
                    if folder_path.exists():
                        file_count = sum(1 for f in folder_path.rglob("*") if f.is_file())
                        if file_count > 0:
                            shutil.rmtree(folder_path)
                            deleted_files += file_count
                            deleted_folders += 1
                            log_message(f"🗑️ Deleted: {module_name}/{folder} ({file_count} files)")

        # Clean up CustomModel
        custom_model_dir = get_pipeline_file("CustomModel")
        if custom_model_dir.exists():
            file_count = sum(1 for f in custom_model_dir.rglob("*") if f.is_file())
            if file_count > 0:
                shutil.rmtree(custom_model_dir)
                deleted_files += file_count
                deleted_folders += 1
                log_message(f"🗑️ Deleted: CustomModel ({file_count} files)")

        if deleted_files == 0:
            return "✅ No files found for cleanup"

        log_message(f"✅ Cleanup completed: {deleted_files} files in {deleted_folders} folders deleted")
        return f"✅ Cleanup successful!\n\n📊 Deleted:\n- {deleted_files} files\n- {deleted_folders} folders"

    except Exception as e:
        error_msg = f"❌ Error during cleanup: {e}"
        log_message(error_msg, "ERROR")
        return error_msg

# ==================== MAIN INTERFACE ====================

def create_main_interface():
    """Creates the main interface."""
    
    # Initialize pipeline status
    reset_pipeline_status()
    initial_overview = create_pipeline_overview()
    
    with gr.Blocks(
        title="OpenTuneWeaver - Finetuning Pipeline UI",
        theme=gr.themes.Soft(),
        css="""
        .header-gradient {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 15px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .terminal-output {
            font-family: 'Courier New', monospace !important;
            background-color: #1a1a1a !important;
            color: #00ff00 !important;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #333;
            height: 600px !important;
            max-height: 600px !important;
            overflow-y: scroll !important;
            overflow-x: hidden;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .status-info {
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #b8d4f2;
        }
        .quick-settings {
            background-color: #f9fafb;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            margin: 10px 0;
        }
        .help-section {
            background-color: #fefce8;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #fde68a;
            margin: 10px 0;
        }
        """
    ) as interface:

        # Header
        gr.HTML("""
        <div class="header-gradient">
            <div style="font-size: 4em; margin-bottom: 10px;">🎯</div>
            <h1 style="margin: 0; font-size: 3em; font-weight: bold;">OpenTuneWeaver</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.95;">
                Your All-In-One Solution for Bringing Your Documents to Your LLM
            </p>
        </div>
        """)

        with gr.Tabs():

            # ==================== HOME PAGE ====================
            with gr.TabItem("🏠 Home"):
                # Pipeline Status Overview
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 🎯 Pipeline Status")
                        with gr.Row():
                            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
                        pipeline_overview_display = gr.HTML(value=initial_overview)
                        
                with gr.Row():
                    with gr.Column(scale=1):
                        # Step Selection
                        gr.Markdown("### 📋 Pipeline Steps")
                        step_checkboxes = gr.CheckboxGroup(
                            choices=[
                                "1. Document Conversion",
                                "2. Wiki Generation",
                                "3. QA Creation",
                                "4. Dataset Formatting",
                                "5. Benchmark Creation",
                                "6. Fine-tuning",
                                "7. Benchmarking",
                                "8. Results Archive"
                            ],
                            value=[
                                "1. Document Conversion",
                                "2. Wiki Generation",
                                "3. QA Creation",
                                "4. Dataset Formatting",
                                "5. Benchmark Creation",
                                "6. Fine-tuning",
                                "7. Benchmarking",
                                "8. Results Archive"
                            ],
                            label="Select steps to run:",
                            interactive=True
                        )
                        step_status = gr.Textbox(
                            label="Selected Steps",
                            value="✅ All steps selected (1-8)",
                            interactive=False,
                            lines=1,
                            elem_classes=["status-info"]
                        )
                        
                        # Quick Settings
                        gr.Markdown("### ⚡ Quick Settings")
                        with gr.Group(elem_classes=["quick-settings"]):
                            model_name_quick = gr.Textbox(
                                label="1. Model Name",
                                value="OpenTuneWeaver-Model",
                                placeholder="Name for your fine-tuned model"
                            )
                            
                            hf_token_quick = gr.Textbox(
                                label="2. HuggingFace Token (Read)",
                                type="password",
                                placeholder="hf_... (required for model downloads)"
                            )
                            
                            preset_dropdown = gr.Dropdown(
                                label="3. Fine-tuning Preset",
                                choices=["Test", "Development", "Production", "Expert"],
                                value="Production",
                                interactive=True
                            )
                            
                            preset_info = gr.Markdown(get_preset_info("Production"))
                            
                            gr.Markdown("**4. Save Options:**")
                            with gr.Row():
                                save_lora_quick = gr.Checkbox(label="LoRA Adapter", value=True)
                                save_merged_quick = gr.Checkbox(label="Merged Model", value=True)
                                save_gguf_quick = gr.Checkbox(label="GGUF Format", value=False)
                            
                            save_quick_btn = gr.Button("💾 Save Settings", variant="primary")
                            quick_status = gr.Textbox(
                                label="Status",
                                interactive=False,
                                elem_classes=["status-info"]
                            )
                        
                        # File Upload
                        gr.Markdown("### 📁 Document Upload")
                        file_upload = gr.File(
                            label="Select files (PDF, DOCX, TXT, MD, etc.)",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".md", ".html", ".xml", ".pptx", ".xlsx", ".rtf", ".odt"]
                        )
                        upload_btn = gr.Button("📤 Upload files", variant="primary")
                        upload_status = gr.Textbox(
                            label="Upload Status",
                            interactive=False,
                            elem_classes=["status-info"]
                        )

                    with gr.Column(scale=1):
                        # Pipeline Control
                        gr.Markdown("### 🚀 Pipeline Control")
                        gr.Markdown("*Fully automated execution*")
                        with gr.Row():
                            start_btn = gr.Button("🚀 Start Pipeline", variant="primary", size="lg")
                            stop_btn = gr.Button("⏹️ Stop Pipeline", variant="stop")
                        
                        pipeline_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            elem_classes=["status-info"]
                        )

                        # Downloads
                        gr.Markdown("### 📥 Downloads")
                        with gr.Row():
                            download_docs_btn = gr.Button("📄 Documents", variant="secondary")
                            download_model_btn = gr.Button("🤖 Model", variant="secondary")
                        
                        download_status = gr.Textbox(
                            label="Download Status",
                            interactive=False,
                            elem_classes=["status-info"]
                        )
                        
                        download_file = gr.File(
                            label="Download File",
                            visible=False
                        )

            # ==================== TERMINAL ====================
            with gr.TabItem("🖥️ Terminal"):
                gr.Markdown("### 🖥️ Live Terminal Output")
                gr.Markdown("*Real-time pipeline progress and logs*")
                with gr.Row():
                    terminal_refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")

                terminal_output = gr.Textbox(
                    label="Terminal Output",
                    value="📋 OpenTuneWeaver Pipeline ready...\n🚀 Configure settings and start the pipeline",
                    interactive=False,
                    lines=30,
                    max_lines=30,
                    elem_classes=["terminal-output"]
                )

            # ==================== EXPERT SETTINGS ====================
            with gr.TabItem("🔬 Expert Settings"):
                gr.Markdown("### 🔧 Expert Configuration")
                gr.Markdown("Advanced settings for experienced users. For beginners, use the Quick Settings on the Home page.")

                with gr.Accordion("🔑 HuggingFace Tokens", open=True):
                    with gr.Row():
                        hf_token = gr.Textbox(
                            label="HF Token (for model downloads)",
                            type="password",
                            placeholder="hf_..."
                        )
                        hf_write_token = gr.Textbox(
                            label="HF Write Token (optional)",
                            type="password",
                            placeholder="hf_..."
                        )

                with gr.Accordion("🌐 API Configuration", open=False):
                    with gr.Row():
                        api_base_url = gr.Textbox(
                            label="API Base URL",
                            value="http://localhost:11434/v1",
                            placeholder="http://localhost:11434/v1"
                        )
                        api_key = gr.Textbox(
                            label="API Key",
                            value="ollama",
                            placeholder="ollama"
                        )

                    gr.Markdown("**Model configuration for each step:**")
                    with gr.Row():
                        convert_model = gr.Textbox(label="📄 Convert Model", value="gemma3:12b-it-qat")
                        convert_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.1, step=0.1)

                    with gr.Row():
                        wiki_model = gr.Textbox(label="📚 Wiki Model", value="gemma3:12b-it-qat")
                        wiki_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.3, step=0.1)

                    with gr.Row():
                        qa_model = gr.Textbox(label="❓ QA Model", value="gemma3:12b-it-qat")
                        qa_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.7, step=0.1)

                    with gr.Row():
                        benchmark_model = gr.Textbox(label="📊 Benchmark Model", value="gemma3:12b-it-qat")
                        benchmark_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, value=0.5, step=0.1)

                with gr.Accordion("🤖 Fine-tuning Configuration", open=False):
                    with gr.Row():
                        model_name = gr.Textbox(label="Model Name", value="OpenTuneWeaver-Model")
                        base_model = gr.Dropdown(
                            label="Base Model",
                            choices=AVAILABLE_MODELS,
                            value="unsloth/gemma-3n-E2B-it",
                            interactive=True
                        )

                    with gr.Row():
                        hf_repo_id = gr.Textbox(label="HuggingFace Repo ID", value="user/OpenTuneWeaver-Model")
                        custom_model_dir = gr.Textbox(label="CustomModel Directory", value="CustomModel")

                    gr.Markdown("**Training Parameters:**")
                    with gr.Row():
                        max_seq_length = gr.Slider(label="Max Sequence Length", minimum=512, maximum=16384, value=8192, step=512)
                        load_in_4bit = gr.Checkbox(label="Load in 4-bit", value=True)
                        full_finetuning = gr.Checkbox(label="Full Fine-tuning", value=False)

                    gr.Markdown("**LoRA Parameters:**")
                    with gr.Row():
                        lora_r = gr.Slider(label="LoRA r", minimum=1, maximum=64, value=8, step=1)
                        lora_alpha = gr.Slider(label="LoRA Alpha", minimum=1, maximum=64, value=8, step=1)
                        lora_dropout = gr.Slider(label="LoRA Dropout", minimum=0.0, maximum=0.5, value=0.0, step=0.05)

                    gr.Markdown("**Training Settings:**")
                    with gr.Row():
                        batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=16, value=1, step=1)
                        grad_accumulation = gr.Slider(label="Gradient Accumulation Steps", minimum=1, maximum=64, value=16, step=1)

                    with gr.Row():
                        warmup_steps = gr.Slider(label="Warmup Steps", minimum=0, maximum=1000, value=200, step=10)
                        num_epochs = gr.Slider(label="Number of Epochs", minimum=1, maximum=20, value=3, step=1)

                    with gr.Row():
                        learning_rate = gr.Slider(label="Learning Rate", minimum=1e-6, maximum=1e-3, value=5e-5, step=1e-6)
                        weight_decay = gr.Slider(label="Weight Decay", minimum=0.0, maximum=0.3, value=0.03, step=0.01)

                    gr.Markdown("**Output Options:**")
                    with gr.Row():
                        save_lora = gr.Checkbox(label="Save LoRA Adapter", value=True)
                        save_merged = gr.Checkbox(label="Save Merged Model", value=True)
                        save_gguf = gr.Checkbox(label="Save GGUF Model", value=False)

                with gr.Accordion("🏆 Benchmark Configuration", open=False):
                    with gr.Row():
                        benchmark_mode = gr.Dropdown(
                            label="Benchmark Mode",
                            choices=["comparison", "post_only", "pre_only"],
                            value="comparison"
                        )
                        evaluator_model = gr.Textbox(label="Evaluator Model", value="gemma3:12b-it-qat")

                    with gr.Row():
                        max_new_tokens = gr.Slider(label="Max New Tokens", minimum=50, maximum=1000, value=256, step=10)
                        eval_temp = gr.Slider(label="Evaluation Temperature", minimum=0.0, maximum=2.0, value=0.3, step=0.1)

                    with gr.Row():
                        top_p = gr.Slider(label="Top P", minimum=0.1, maximum=1.0, value=0.9, step=0.05)
                        top_k = gr.Slider(label="Top K", minimum=1, maximum=100, value=50, step=1)

                    repetition_penalty = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, value=1.1, step=0.05)

                with gr.Accordion("⚙️ Pipeline Settings", open=False):
                    with gr.Row():
                        auto_cleanup = gr.Checkbox(label="Automatic Cleanup", value=False)
                        verbose_mode = gr.Checkbox(label="Verbose Mode", value=True)
                        continue_on_error = gr.Checkbox(label="Continue on Error", value=True)

                # Buttons
                with gr.Row():
                    load_config_btn = gr.Button("📋 Load Current Configuration", variant="secondary")
                    save_config_btn = gr.Button("💾 Save Configuration", variant="primary")

                config_status = gr.Textbox(
                    label="Configuration Status",
                    interactive=False,
                    elem_classes=["status-info"]
                )

            # ==================== CLEANUP ====================
            with gr.TabItem("🗑️ Cleanup"):
                with gr.Column():
                    gr.Markdown("### 🧹 Clean up pipeline directories")
                    gr.Markdown("Delete all temporary files and results for a clean restart")

                    with gr.Row():
                        with gr.Column(scale=1):
                            analyze_btn = gr.Button("🔍 Analyze directories", variant="secondary")
                            cleanup_confirm_btn = gr.Button("🗑️ DELETE ALL", variant="stop")

                            cleanup_result = gr.Textbox(
                                label="Cleanup Result",
                                interactive=False,
                                elem_classes=["status-info"]
                            )

                        with gr.Column(scale=2):
                            cleanup_info = gr.Textbox(
                                label="Cleanup Analysis",
                                value="📊 Click 'Analyze directories' to see what would be deleted",
                                interactive=False,
                                lines=20
                            )

                    gr.Markdown("""
                    ### ⚠️ Important Notes
                    
                    **What will be deleted:**
                    - 📁 All INPUT/OUTPUT directories
                    - 📁 Uploaded documents
                    - 📁 Trained models
                    - 📁 Generated datasets
                    - 📁 Benchmark results
                    
                    **⚠️ This action CANNOT be undone!**
                    **Download your results before cleanup!**
                    """)

            # ==================== HELP PAGE ====================
            with gr.TabItem("📚 Help"):
                gr.Markdown("### 📖 OpenTuneWeaver Guide")
                
                with gr.Accordion("🎯 Quick Start", open=True):
                    gr.Markdown("""
                    **Getting started in 5 steps:**
                    
                    1. **Upload Documents**: Upload your training documents (PDF, DOCX, TXT, MD)
                    2. **Set Model Name**: Choose a name for your fine-tuned model
                    3. **Add HF Token**: Enter your HuggingFace token for model downloads
                    4. **Select Preset**: Choose Test, Development, or Production
                    5. **Start Pipeline**: Click "Start Pipeline" and wait for completion
                    
                    **Example:**
                    - Model Name: `MyCustomAssistant`
                    - HF Token: `hf_AbCdEfGhIjKlMnOpQrStUvWxYz`
                    - Preset: `Development` (balanced speed/quality)
                    - Save Options: ✅ LoRA, ✅ Merged Model
                    """)
                
                with gr.Accordion("🔧 Presets Explained", open=True):
                    gr.Markdown("""
                    ### Fine-tuning Presets
                    
                    **🧪 Test Preset**
                    - Purpose: Quick testing and validation
                    - Training Time: ~15-30 minutes
                    - Quality: Basic, for testing pipeline
                    - Settings: 1 epoch, 2048 seq length, LoRA r=4
                    - Use When: Testing new documents or pipeline setup
                    
                    **🔨 Development Preset**
                    - Purpose: Development and iteration
                    - Training Time: ~1-2 hours
                    - Quality: Good, suitable for development
                    - Settings: 2 epochs, 4096 seq length, LoRA r=8
                    - Use When: Developing and testing your model
                    
                    **🏭 Production Preset**
                    - Purpose: Final production models
                    - Training Time: ~2-4 hours
                    - Quality: Best possible results
                    - Settings: 3 epochs, 8192 seq length, LoRA r=16
                    - Use When: Creating final production models
                    
                    **🔬 Expert Preset**
                    - Purpose: Custom configuration
                    - Use the Expert Settings page for full control
                    """)
                
                with gr.Accordion("📋 Pipeline Steps Explained", open=False):
                    gr.Markdown("""
                    ### What each step does:
                    
                    **1. Document Conversion** 📄
                    - Converts uploaded files to clean text
                    - Supports PDF, DOCX, TXT, MD, HTML, etc.
                    - Removes formatting and extracts content
                    
                    **2. Wiki Generation** 📚
                    - Creates Wikipedia-style articles from documents
                    - Generates structured knowledge base
                    - Improves model understanding
                    
                    **3. QA Creation** ❓
                    - Generates question-answer pairs
                    - Creates training examples
                    - Builds comprehension tests
                    
                    **4. Dataset Formatting** 🔧
                    - Formats data for training
                    - Creates JSON training file
                    - Validates data structure
                    
                    **5. Benchmark Creation** 📊
                    - Creates evaluation questions
                    - Builds test dataset
                    - Prepares comparison metrics
                    
                    **6. Fine-tuning** 🤖
                    - Trains the model on your data
                    - Applies LoRA adapters
                    - Creates custom model
                    
                    **7. Benchmarking** 🏆
                    - Tests model performance
                    - Compares before/after
                    - Generates metrics
                    
                    **8. Results Archive** 📦
                    - Packages all results
                    - Creates downloadable archive
                    - Saves training history
                    """)
                
                with gr.Accordion("💡 Tips & Tricks", open=False):
                    gr.Markdown("""
                    ### Best Practices:
                    
                    **Document Preparation:**
                    - Use high-quality, clean documents
                    - Mix different types of content
                    - Include at least 10-20 pages of text
                    - Remove unnecessary headers/footers
                    
                    **Model Selection:**
                    - Start with smaller models (E2B) for testing
                    - Use larger models (E4B) for production
                    - Consider VRAM requirements
                    
                    **Training Tips:**
                    - Start with Test preset to verify setup
                    - Use Development for iterating
                    - Only use Production for final models
                    - Monitor terminal for progress
                    
                    **Troubleshooting:**
                    - Check terminal output for errors
                    - Ensure Ollama is running for API calls
                    - Verify HuggingFace token is valid
                    - Use Cleanup tab if restarting
                    """)

        # ==================== EVENT HANDLERS ====================

        # Step selection handler
        step_checkboxes.change(
            fn=update_step_selection,
            inputs=[step_checkboxes],
            outputs=[step_status]
        )

        # Refresh status handler
        refresh_btn.click(
            fn=lambda: create_pipeline_overview(),
            outputs=[pipeline_overview_display]
        )

        # Quick Settings handlers
        preset_dropdown.change(
            fn=lambda x: get_preset_info(x),
            inputs=[preset_dropdown],
            outputs=[preset_info]
        )

        save_quick_btn.click(
            fn=save_config_from_quick_settings,
            inputs=[
                model_name_quick, hf_token_quick, preset_dropdown,
                save_lora_quick, save_merged_quick, save_gguf_quick
            ],
            outputs=[quick_status]
        )

        # Upload handler
        upload_btn.click(
            fn=save_uploaded_files,
            inputs=[file_upload],
            outputs=[upload_status]
        )

        # Pipeline handlers
        start_btn.click(
            fn=start_pipeline,
            outputs=[pipeline_status, pipeline_overview_display]
        )

        stop_btn.click(
            fn=stop_pipeline,
            outputs=[pipeline_status, pipeline_overview_display]
        )

        # Terminal handlers
        terminal_refresh_btn.click(
            fn=get_terminal_output,
            outputs=[terminal_output]
        )

        clear_btn.click(
            fn=clear_terminal,
            outputs=[terminal_output]
        )

        # Download handlers
        def handle_documents_download():
            zip_path, status = create_documents_zip()
            if zip_path:
                return status, zip_path, gr.update(visible=True)
            else:
                return status, None, gr.update(visible=False)

        def handle_model_download():
            zip_path, status = create_model_zip()
            if zip_path:
                return status, zip_path, gr.update(visible=True)
            else:
                return status, None, gr.update(visible=False)

        download_docs_btn.click(
            fn=handle_documents_download,
            outputs=[download_status, download_file, download_file]
        )

        download_model_btn.click(
            fn=handle_model_download,
            outputs=[download_status, download_file, download_file]
        )

        # Cleanup handlers
        analyze_btn.click(
            fn=get_cleanup_info,
            outputs=[cleanup_info]
        )

        cleanup_confirm_btn.click(
            fn=cleanup_pipeline_folders,
            outputs=[cleanup_result]
        )

        # Expert Configuration handlers
        def load_config_to_ui():
            config = load_existing_config()
            
            # Extract all values from config
            tokens = config.get("tokens", {})
            hf_token_val = tokens.get("hf_token", "")
            hf_write_token_val = tokens.get("hf_write_token", "")
            
            # API Config
            api_config = config.get("api_configs", {}).get("01_convert", {})
            api_base_url_val = api_config.get("openai_base_url", "http://localhost:11434/v1")
            api_key_val = api_config.get("openai_api_key", "ollama")
            
            # Models
            convert_config = config.get("api_configs", {}).get("01_convert", {})
            wiki_config = config.get("api_configs", {}).get("02_genwiki", {})
            qa_config = config.get("api_configs", {}).get("03_instructQA", {})
            benchmark_config = config.get("api_configs", {}).get("05_bmcreator", {})
            
            # Fine-tuning
            ft_config = config.get("finetuning", {})
            
            # Benchmark
            bench_config = config.get("benchmark", {})
            
            # Pipeline
            pipe_config = config.get("pipeline", {})
            
            return [
                hf_token_val, hf_write_token_val,
                api_base_url_val, api_key_val,
                convert_config.get("openai_model_name", "gemma3:12b-it-qat"),
                wiki_config.get("openai_model_name", "gemma3:12b-it-qat"),
                qa_config.get("openai_model_name", "gemma3:12b-it-qat"),
                benchmark_config.get("openai_model_name", "gemma3:12b-it-qat"),
                convert_config.get("temperature", 0.1),
                wiki_config.get("temperature", 0.3),
                qa_config.get("temperature", 0.7),
                benchmark_config.get("temperature", 0.5),
                ft_config.get("model_name", "OpenTuneWeaver-Model"),
                ft_config.get("base_model", "unsloth/gemma-3n-E2B-it"),
                ft_config.get("hf_repo_id", "user/OpenTuneWeaver-Model"),
                ft_config.get("custom_model_dir", "CustomModel"),
                ft_config.get("max_seq_length", 8192),
                ft_config.get("load_in_4bit", True),
                ft_config.get("full_finetuning", False),
                ft_config.get("lora_r", 8),
                ft_config.get("lora_alpha", 8),
                ft_config.get("lora_dropout", 0.0),
                ft_config.get("per_device_train_batch_size", 1),
                ft_config.get("gradient_accumulation_steps", 16),
                ft_config.get("warmup_steps", 200),
                ft_config.get("num_train_epochs", 3),
                ft_config.get("learning_rate", 5e-5),
                ft_config.get("weight_decay", 0.03),
                ft_config.get("save_lora", True),
                ft_config.get("save_merged", True),
                ft_config.get("save_gguf", False),
                bench_config.get("mode", "comparison"),
                bench_config.get("evaluator", {}).get("model", "gemma3:12b-it-qat"),
                bench_config.get("max_new_tokens", 256),
                bench_config.get("temperature", 0.3),
                bench_config.get("top_p", 0.9),
                bench_config.get("top_k", 50),
                bench_config.get("repetition_penalty", 1.1),
                pipe_config.get("auto_cleanup", False),
                pipe_config.get("verbose", True),
                pipe_config.get("continue_on_error", True),
                "✅ Configuration loaded"
            ]

        load_config_btn.click(
            fn=load_config_to_ui,
            outputs=[
                hf_token, hf_write_token,
                api_base_url, api_key,
                convert_model, wiki_model, qa_model, benchmark_model,
                convert_temp, wiki_temp, qa_temp, benchmark_temp,
                model_name, base_model, hf_repo_id, custom_model_dir,
                max_seq_length, load_in_4bit, full_finetuning,
                lora_r, lora_alpha, lora_dropout,
                batch_size, grad_accumulation, warmup_steps, num_epochs,
                learning_rate, weight_decay,
                save_lora, save_merged, save_gguf,
                benchmark_mode, evaluator_model, max_new_tokens,
                eval_temp, top_p, top_k, repetition_penalty,
                auto_cleanup, verbose_mode, continue_on_error,
                config_status
            ]
        )

        save_config_btn.click(
            fn=save_config_from_ui,
            inputs=[
                hf_token, hf_write_token,
                api_base_url, api_key,
                convert_model, wiki_model, qa_model, benchmark_model,
                convert_temp, wiki_temp, qa_temp, benchmark_temp,
                model_name, base_model, hf_repo_id, custom_model_dir,
                max_seq_length, load_in_4bit, full_finetuning,
                lora_r, lora_alpha, lora_dropout,
                batch_size, grad_accumulation, warmup_steps, num_epochs,
                learning_rate, weight_decay,
                save_lora, save_merged, save_gguf,
                benchmark_mode, evaluator_model, max_new_tokens,
                eval_temp, top_p, top_k, repetition_penalty,
                auto_cleanup, verbose_mode, continue_on_error
            ],
            outputs=[config_status]
        )

        # Auto-refresh for pipeline overview and terminal
        def auto_refresh():
            overview = create_pipeline_overview()
            terminal = get_terminal_output()
            return overview, terminal

        try:
            interface.load(
                fn=auto_refresh,
                outputs=[pipeline_overview_display, terminal_output],
                every=3  # Update every 3 seconds
            )
            log_message("✅ Auto-refresh activated (3 seconds)")
        except:
            log_message("⚠️ Auto-refresh not supported - use 'Refresh' button")

        return interface

def main():
    """Main function."""
    print("="*80)
    print(" 🎯 OPENTUNEWEAVER - FINETUNING PIPELINE UI v11.0")
    print("="*80)
    
    # Show configuration
    print(f"📁 Project Root: {PROJECT_ROOT}")
    print(f"📁 Pipeline Path: {PIPELINE_PATH}")
    print(f"📁 UI Path: {UI_PATH}")
    print(f"🌐 Environment: {'RunPod' if IS_RUNPOD else 'Local'}")
    
    log_message("🚀 Starting OpenTuneWeaver UI...")
    log_message("✨ Enhanced with Expert Settings and Help Page")

    # Create interface
    interface = create_main_interface()

    # Start server
    log_message("🌐 Starting web server...")
    print("\n🌐 UI available at:")
    print(" - Local: http://localhost:8080")
    print(" - Network: http://YOUR_IP:8080")
    print("\n🎯 Features:")
    print(" - 📁 Direct access to pipeline and ui folders")
    print(" - 🔬 Expert Settings for advanced configuration")
    print(" - 📚 Comprehensive Help documentation")
    print(" - 📊 Auto-updating pipeline status")
    print(" - ✅ Step selection (run individual steps)")
    print(" - 📱 Mobile-responsive design")

    try:
        interface.launch(
            server_name="0.0.0.0",
            server_port=8080,
            share=False,
            debug=False,
            show_error=True,
            quiet=False,
            inbrowser=False
        )
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ OpenTuneWeaver UI terminated")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()