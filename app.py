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

FINETUNING_PRESETS = {}

# Global variables
progress_messages: List[str] = []
processing_active: bool = False
current_process: Optional[subprocess.Popen] = None
current_preset: Optional[str] = None
pipeline_status: Dict[int, Dict] = {}
selected_steps: List[int] = [1, 2, 3]  # Default: all steps

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

def open_viewer():
    """Opens the local otw_dataeditor.html viewer in the default browser."""
    import webbrowser
    import os
    viewer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "viewer", "otw_dataeditor.html"))
    if os.path.exists(viewer_path):
        webbrowser.open(f"file://{viewer_path}")
        return "✅ Viewer opened in your browser"
    return "❌ Viewer file not found at viewer/otw_dataeditor.html"

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

    saved_count: int = 0
    for file in files:
        if file is not None:
            try:
                file_path = Path(file.name)
                target_path = upload_dir / file_path.name
                shutil.copy2(file_path, target_path)
                log_message(f"✅ File saved: {file_path.name}")
                saved_count = saved_count + 1
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
        3: {"name": "Generate QA Dataset", "status": "pending", "icon": "❓", "stats": {}, "duration": None}
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
                        if int_id in range(1, 4):
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
    
    for step_id in range(1, 4):
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
                stats_text += f"📁 {stats.get('files_processed', 0)} files<br>"
            if "entries" in stats:
                stats_text += f"📝 {stats.get('entries', 0)} entries<br>"
        
            if isinstance(duration, (int, float)):
                if duration < 60:
                    duration_text = f"⏱️ {duration:.0f}s"
                else:
                    duration_text = f"⏱️ {duration/60:.1f}m"
                stats_text += duration_text
        
        status_icon_upper = status.upper() if isinstance(status, str) else "PENDING"
        card_html = f"""
        <div style="background: white; border-radius: 8px; padding: 10px; text-align: center; 
                    border: {border_style}; position: relative; min-height: 120px;">
            <div style="font-size: 1.8em; margin-bottom: 3px;">{icon}</div>
            <div style="color: #1f2937; font-weight: bold; font-size: 0.85em;">Step {step_id}</div>
            <div style="color: #4b5563; font-size: 0.75em; margin: 3px 0;">{name}</div>
            <div style="background: {bg_color}; color: white; padding: 2px 6px; border-radius: 4px; 
                        font-size: 0.7em; margin: 5px 0;">
                {status_icon} {status_icon_upper}
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
    total = 3
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
    
    if len(selected_steps) == 3:
        return "✅ All steps selected (1-3)"
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
            "03_generate_qa": {
                "use_openai_api": True,
                "openai_base_url": "http://localhost:11434/v1",
                "openai_api_key": "ollama",
                "openai_model_name": "gemma3:12b-it-qat",
                "temperature": 0.7
            }
        },
        "pipeline": {
            "auto_cleanup": False,
            "verbose": True,
            "continue_on_error": True,
            "save_metrics": True
        }
    }

def save_config_from_ui(
    # API Configs
    api_base_url, api_key,
    convert_model, wiki_model, qa_model,
    convert_temp, wiki_temp, qa_temp,
    # Pipeline
    auto_cleanup, verbose_mode, continue_on_error
):
    """Saves the configuration from the UI.
    
    IMPORTANT: config_loader.py reads 'vision' and 'llm' top-level keys.
    01_convert uses vision_config, 02_genwiki and 03_generate_qa use llm_config.
    We write both the unified keys AND per-module api_configs for compatibility.
    """
    try:
        config = {
            "version": "12.0",
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            # === Unified keys read by config_loader.py ===
            "vision": {
                "api_base_url": api_base_url,
                "api_key": api_key,
                "model_name": convert_model,
                "temperature": convert_temp
            },
            "llm": {
                "api_base_url": api_base_url,
                "api_key": api_key,
                "model_name": wiki_model,
                "temperature": wiki_temp
            },
            # === Per-module overrides (temperatures, model names) ===
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
                "03_generate_qa": {
                    "use_openai_api": True,
                    "openai_base_url": api_base_url,
                    "openai_api_key": api_key,
                    "openai_model_name": qa_model,
                    "temperature": qa_temp
                }
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
    
    if steps == [1, 2, 3]:
        return "full"
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
            if current_process and current_process.stdout:
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
        if current_process:
            # Try graceful termination
            current_process.terminate()
            time.sleep(2)
            
            # Check current process is completely terminated
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
                    ("03_generate_qa", ["OUTPUT", "INPUT"])
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
            
            # CustomModel directory logic disabled because model outputs are removed.
            
            for custom_model_dir in []:
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

        total_files: int = 0
        total_size: int = 0
        cleanup_info: List[str] = []
        
        # Module folders
        module_folders = [
            ("01_convert", ["INPUT", "OUTPUT", "UPLOAD"]),
            ("02_wiki", ["INPUT", "OUTPUT"]),
            ("03_generate_qa", ["INPUT", "OUTPUT"])
        ]
        
        for module_name, folders in module_folders:
            module_dir = modules_dir / module_name
            if module_dir.exists():
                for folder in folders:
                    folder_path = module_dir / folder
                    if folder_path.exists():
                        file_count: int = 0
                        folder_size: int = 0
                        for file_path in folder_path.rglob("*"):
                            if file_path.is_file():
                                file_count = file_count + 1
                                folder_size = folder_size + file_path.stat().st_size
                        
                        if file_count > 0:
                            size_mb = folder_size / (1024 * 1024)
                            cleanup_info.append(f"📁 {module_name}/{folder}: {file_count} files ({size_mb:.1f} MB)")
                            total_files = total_files + file_count
                            total_size = total_size + folder_size

        if not cleanup_info:
            return "✅ No files found for cleanup"

        total_size_mb: float = float(total_size) / (1024 * 1024)
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

        deleted_files: int = 0
        deleted_folders: int = 0

        module_folders = [
            ("01_convert", ["INPUT", "OUTPUT", "UPLOAD"]),
            ("02_wiki", ["INPUT", "OUTPUT"]),
            ("03_generate_qa", ["INPUT", "OUTPUT"])
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
                            deleted_files = deleted_files + file_count
                            deleted_folders = deleted_folders + 1
                            log_message(f"🗑️ Deleted: {module_name}/{folder} ({file_count} files)")

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
    
    # Build logo path for serving
    logo_path = str(PROJECT_ROOT / "assets" / "otw_logo.png")
    
    with gr.Blocks(
        title="OpenTuneWeaver - Finetuning Pipeline UI",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#111827",
            body_background_fill_dark="#111827",
            body_text_color="#e5e7eb",
            body_text_color_dark="#e5e7eb",
            background_fill_primary="#1f2937",
            background_fill_primary_dark="#1f2937",
            background_fill_secondary="#374151",
            background_fill_secondary_dark="#374151",
            block_background_fill="#1f2937",
            block_background_fill_dark="#1f2937",
            block_border_color="#374151",
            block_border_color_dark="#374151",
            block_label_background_fill="#374151",
            block_label_background_fill_dark="#374151",
            block_label_text_color="#d1d5db",
            block_label_text_color_dark="#d1d5db",
            block_title_text_color="#f9fafb",
            block_title_text_color_dark="#f9fafb",
            border_color_primary="#4b5563",
            border_color_primary_dark="#4b5563",
            input_background_fill="#374151",
            input_background_fill_dark="#374151",
            input_border_color="#4b5563",
            input_border_color_dark="#4b5563",
            button_primary_background_fill="#0d9488",
            button_primary_background_fill_dark="#0d9488",
            button_primary_background_fill_hover="#0f766e",
            button_primary_background_fill_hover_dark="#0f766e",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill="#374151",
            button_secondary_background_fill_dark="#374151",
            button_secondary_text_color="#d1d5db",
            button_secondary_text_color_dark="#d1d5db",
            panel_background_fill="#1f2937",
            panel_background_fill_dark="#1f2937",
        ),
        css="""
        .gradio-container {
            max-width: 100% !important;
        }
        .otw-header {
            background: #1f2937;
            padding: 16px 24px;
            margin-bottom: 12px;
            border-radius: 8px;
            border: 1px solid #374151;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .otw-header img {
            height: 48px;
            width: auto;
        }
        .otw-header .otw-title {
            font-size: 1.5em;
            font-weight: 700;
            color: #f9fafb;
            letter-spacing: -0.02em;
        }
        .otw-header .otw-subtitle {
            font-size: 0.85em;
            color: #9ca3af;
            margin-top: 2px;
        }
        .terminal-output {
            font-family: 'Courier New', monospace !important;
            background-color: #0f172a !important;
            color: #4ade80 !important;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #1e293b;
            height: 600px !important;
            max-height: 600px !important;
            overflow-y: scroll !important;
            overflow-x: hidden;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .status-info {
            background-color: #1e293b;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #334155;
            color: #94a3b8;
        }
        .quick-settings {
            background-color: #1f2937;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #374151;
            margin: 10px 0;
        }
        .help-section {
            background-color: #1c1917;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #44403c;
            margin: 10px 0;
            color: #d6d3d1;
        }
        /* Tab styling */
        .tabs > .tab-nav > button {
            color: #9ca3af !important;
            border-color: #374151 !important;
        }
        .tabs > .tab-nav > button.selected {
            color: #14b8a6 !important;
            border-color: #14b8a6 !important;
        }
        /* Accordion */
        .label-wrap {
            color: #d1d5db !important;
        }
        /* Make textboxes readable */
        textarea, input[type="text"], input[type="password"], input[type="number"] {
            color: #e5e7eb !important;
            background-color: #374151 !important;
            border-color: #4b5563 !important;
        }
        """
    ) as interface:

        # Header with logo
        import base64
        logo_b64 = ""
        try:
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            pass
        
        if logo_b64:
            gr.HTML(f"""
            <div class="otw-header">
                <img src="data:image/png;base64,{logo_b64}" alt="OTW Logo">
                <div>
                    <div class="otw-title">OpenTuneWeaver</div>
                    <div class="otw-subtitle">Your All-In-One Solution for Bringing Your Documents to Your LLM</div>
                </div>
            </div>
            """)
        else:
            gr.HTML("""
            <div class="otw-header">
                <div>
                    <div class="otw-title">OpenTuneWeaver</div>
                    <div class="otw-subtitle">Your All-In-One Solution for Bringing Your Documents to Your LLM</div>
                </div>
            </div>
            """)

        with gr.Tabs():

            # ==================== HOME PAGE ====================
            with gr.TabItem("🏠 Home"):
                # Pipeline Status Overview
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 🎯 Output Hub & Status")
                        with gr.Row():
                            viewer_launch_btn = gr.Button("🖥️ Launch Viewer Hub", variant="primary", size="lg")
                        with gr.Row():
                            viewer_status = gr.Textbox(visible=False)
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
                                "3. Generate QA Dataset"
                            ],
                            value=[
                                "1. Document Conversion",
                                "2. Wiki Generation",
                                "3. Generate QA Dataset"
                            ],
                            label="Select steps to run:",
                            interactive=True
                        )
                        step_status = gr.Textbox(
                            label="Selected Steps",
                            value="✅ All steps selected (1-3)",
                            interactive=False,
                            lines=1,
                            elem_classes=["status-info"]
                        )
                        
                        # Basic UI is ready now
                        
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
                            download_docs_btn = gr.Button("📄 Download Generated Datasets", variant="secondary")
                        
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

                # Removed HF Tokens Accordion

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
                        test_api_btn = gr.Button("🔌 Test Connection", variant="secondary")
                    
                    api_test_result = gr.Textbox(
                        label="Connection Status",
                        interactive=False,
                        visible=False
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
                    - Generates question-answer pairs in Chat-Masterformat directly into JSONL
                    - Creates training examples
                    - Builds comprehension tests
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

        # Viewer launch handler
        viewer_launch_btn.click(
            fn=open_viewer,
            outputs=[viewer_status]
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

        download_docs_btn.click(
            fn=handle_documents_download,
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
            # API Config - read from unified keys first, fallback to api_configs
            vision = config.get("vision", {})
            llm = config.get("llm", {})
            
            # Fallback: try api_configs if unified keys are empty
            convert_cfg = config.get("api_configs", {}).get("01_convert", {})
            wiki_cfg = config.get("api_configs", {}).get("02_genwiki", {})
            qa_cfg = config.get("api_configs", {}).get("03_generate_qa", {})
            
            api_base_url_val = vision.get("api_base_url") or convert_cfg.get("openai_base_url", "")
            api_key_val = vision.get("api_key") or convert_cfg.get("openai_api_key", "")
            
            convert_model_val = vision.get("model_name") or convert_cfg.get("openai_model_name", "")
            wiki_model_val = llm.get("model_name") or wiki_cfg.get("openai_model_name", "")
            qa_model_val = qa_cfg.get("openai_model_name") or llm.get("model_name", "")
            
            convert_temp_val = convert_cfg.get("temperature", vision.get("temperature", 0.1))
            wiki_temp_val = wiki_cfg.get("temperature", llm.get("temperature", 0.3))
            qa_temp_val = qa_cfg.get("temperature", 0.7)
            
            # Pipeline
            pipe_config = config.get("pipeline", {})
            
            return [
                api_base_url_val, api_key_val,
                convert_model_val,
                wiki_model_val,
                qa_model_val,
                convert_temp_val,
                wiki_temp_val,
                qa_temp_val,
                pipe_config.get("auto_cleanup", False) if isinstance(pipe_config, dict) else False,
                pipe_config.get("verbose", True) if isinstance(pipe_config, dict) else True,
                pipe_config.get("continue_on_error", True) if isinstance(pipe_config, dict) else True,
                "✅ Configuration loaded"
            ]

        load_config_btn.click(
            fn=load_config_to_ui,
            outputs=[
                api_base_url, api_key,
                convert_model, wiki_model, qa_model,
                convert_temp, wiki_temp, qa_temp,
                auto_cleanup, verbose_mode, continue_on_error,
                config_status
            ]
        )

        save_config_btn.click(
            fn=save_config_from_ui,
            inputs=[
                api_base_url, api_key,
                convert_model, wiki_model, qa_model,
                convert_temp, wiki_temp, qa_temp,
                auto_cleanup, verbose_mode, continue_on_error
            ],
            outputs=[config_status]
        )

        # API Test Handler
        def test_api_connection(url: str, key: str, model_name: str) -> tuple[str, dict]:
            import requests
            try:
                if not url.endswith('/'):
                    url = f"{url}/"
                models_endpoint = f"{url}models"
                
                headers = {
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                }
                
                response = requests.get(models_endpoint, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    models_data = response.json()
                    available_models = [m.get('id', m.get('name', 'Unknown')) for m in models_data.get('data', [])]
                    
                    if model_name in available_models or not available_models:
                        return f"✅ Connection successful! Model '{model_name}' is reachable.", gr.update(visible=True)
                    else:
                        return f"⚠️ Connected to API, but '{model_name}' is not in the list of available models. Available: {', '.join(available_models[:5])}...", gr.update(visible=True)
                else:
                    return f"❌ API Error {response.status_code}: {response.text}", gr.update(visible=True)
            except Exception as e:
                return f"❌ Connection failed: {e}", gr.update(visible=True)

        test_api_btn.click(
            fn=test_api_connection,
            inputs=[api_base_url, api_key, convert_model],
            outputs=[api_test_result, api_test_result]
        )

        # Auto-refresh for pipeline overview and terminal
        timer = gr.Timer(3)
        
        def auto_refresh():
            overview = create_pipeline_overview()
            terminal = get_terminal_output()
            return overview, terminal

        timer.tick(
            fn=auto_refresh,
            outputs=[pipeline_overview_display, terminal_output]
        )
        
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