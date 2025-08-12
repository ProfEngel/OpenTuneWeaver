#!/usr/bin/env python3

"""
OpenTuneWeaver UI - Grafische Benutzeroberfläche für die Finetuning-Pipeline
Version: 6.3 - Mit verbesserter Pipeline-Automatisierung

Features:
- Upload-Interface für Dokumente
- Erweiterte Einstellungsseite
- Pipeline-Steuerung über Kommandozeilenparameter
- Live-Terminal mit Auto-Scroll
- Download-Management
- Aufräumen-Funktionalität
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

# Globale Variablen
progress_messages = []
processing_active = False
current_process = None

def log_message(message: str, level: str = "INFO"):
    """Fügt eine Log-Nachricht hinzu."""
    global progress_messages
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {level}: {message}"
    progress_messages.append(formatted_message)
    print(formatted_message)
    return formatted_message

def get_terminal_output():
    """Gibt alle Log-Nachrichten zurück."""
    global progress_messages
    return '\n'.join(progress_messages[-100:])  # Letzte 100 Nachrichten

def clear_terminal():
    """Löscht Terminal-Ausgabe."""
    global progress_messages
    progress_messages = []
    return ""

# ==================== DATEI-UPLOAD ====================

def save_uploaded_files(files: List) -> str:
    """Speichert hochgeladene Dateien in das Upload-Verzeichnis."""
    if not files:
        return "❌ Keine Dateien ausgewählt"

    upload_dir = Path("../pipeline/modules/01_convert/UPLOAD")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Lösche alte Dateien
    for old_file in upload_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink()

    saved_count = 0
    for file in files:
        if file is not None:
            try:
                file_path = Path(file.name)
                target_path = upload_dir / file_path.name
                shutil.copy2(file_path, target_path)
                log_message(f"📁 Datei gespeichert: {file_path.name}")
                saved_count += 1
            except Exception as e:
                log_message(f"❌ Fehler beim Speichern: {e}", "ERROR")

    return f"✅ {saved_count} Dateien erfolgreich hochgeladen"

# ==================== KONFIGURATION ====================

def load_existing_config():
    """Lädt existierende Konfiguration falls vorhanden."""
    config_file = Path("../pipeline/pipeline_config.json")
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            log_message(f"❌ Fehler beim Laden der Konfiguration: {e}", "ERROR")
    
    # Standard-Konfiguration
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
    """Speichert die Konfiguration aus der UI."""
    try:
        config = {
            "version": "6.3",
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

        config_file = Path("../pipeline/pipeline_config.json")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        log_message("✅ Konfiguration gespeichert")
        return "✅ Konfiguration erfolgreich gespeichert!"
        
    except Exception as e:
        error_msg = f"❌ Fehler beim Speichern der Konfiguration: {e}"
        log_message(error_msg, "ERROR")
        return error_msg

# ==================== PIPELINE-STEUERUNG (VERBESSERT) ====================

def start_pipeline(pipeline_mode="full", cleanup_after=True):
    """Startet die Pipeline mit Kommandozeilenparametern."""
    global processing_active, current_process

    if processing_active:
        return "⚠️ Pipeline läuft bereits!"

    def run_pipeline():
        global processing_active, current_process
        processing_active = True
        log_message(f"🚀 Pipeline gestartet im Modus: {pipeline_mode}")

        try:
            # Wechsle ins Pipeline-Verzeichnis
            pipeline_dir = Path("../pipeline")
            if not pipeline_dir.exists():
                log_message("❌ Pipeline-Verzeichnis nicht gefunden!", "ERROR")
                processing_active = False
                return

            original_dir = os.getcwd()
            os.chdir(pipeline_dir)

            # Erstelle Kommandozeilenargumente
            cmd_args = [
                sys.executable, 
                "-u",  # Unbuffered output
                "run_pipeline.py",
                "--auto",  # Automatisierter Modus
                "--mode", pipeline_mode,
                "--use-existing-config"
            ]
            
            if cleanup_after:
                cmd_args.append("--cleanup-after")
            
            log_message(f"🖥️ Kommando: {' '.join(cmd_args)}")

            # Starte run_pipeline.py mit Parametern
            current_process = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Lese Output und zeige im Terminal
            for line in iter(current_process.stdout.readline, ''):
                if not line:
                    break
                
                # Log alle Ausgaben
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

            # Warte auf Pipeline-Ende
            return_code = current_process.wait()
            
            os.chdir(original_dir)

            if return_code == 0:
                log_message("🎉 Pipeline erfolgreich abgeschlossen!")
            else:
                log_message(f"❌ Pipeline fehlgeschlagen (Code: {return_code})", "ERROR")

        except Exception as e:
            log_message(f"❌ Pipeline-Fehler: {e}", "ERROR")
        finally:
            processing_active = False
            current_process = None
            if 'original_dir' in locals():
                os.chdir(original_dir)

    # Starte Pipeline in eigenem Thread
    pipeline_thread = threading.Thread(target=run_pipeline)
    pipeline_thread.daemon = True
    pipeline_thread.start()

    return f"🚀 Pipeline gestartet im Modus '{pipeline_mode}' - Verfolgen Sie den Fortschritt im Terminal"

def start_full_pipeline():
    """Startet die komplette Pipeline (Schritte 1-8)."""
    return start_pipeline("full", cleanup_after=True)

def start_data_pipeline():
    """Startet nur die Datenverarbeitung (Schritte 1-5)."""
    return start_pipeline("data", cleanup_after=False)

def start_training_pipeline():
    """Startet nur Training & Benchmark (Schritte 6-7)."""
    return start_pipeline("training", cleanup_after=False)

def start_archive_only():
    """Startet nur die Archivierung (Schritt 8)."""
    return start_pipeline("archive", cleanup_after=True)

def stop_pipeline():
    """Stoppt die laufende Pipeline."""
    global processing_active, current_process

    if not processing_active or current_process is None:
        return "⚠️ Keine aktive Pipeline gefunden"

    try:
        # Versuche graceful termination
        current_process.terminate()
        time.sleep(2)
        
        # Falls noch aktiv, force kill
        if current_process.poll() is None:
            current_process.kill()
        
        log_message("⏹️ Pipeline gestoppt", "WARNING")
        processing_active = False
        current_process = None
        return "⏹️ Pipeline gestoppt"
    except Exception as e:
        log_message(f"❌ Fehler beim Stoppen: {e}", "ERROR")
        return f"❌ Fehler beim Stoppen: {e}"

# ==================== DOWNLOAD ====================

def create_download_zip():
    """Erstellt eine ZIP-Datei mit allen Ergebnissen."""
    try:
        log_message("📦 Erstelle Download-ZIP...")

        temp_dir = tempfile.mkdtemp()
        zip_path = Path(temp_dir) / f"opentuneweaver_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Suche nach archivierten Ergebnissen in data/OUTPUT
            data_output = Path("../pipeline/data/OUTPUT")
            if data_output.exists():
                # Finde neueste Archive
                archives = list(data_output.glob("*.zip"))
                if archives:
                    # Nimm die neuesten Archive
                    latest_archives = sorted(archives, key=os.path.getctime, reverse=True)[:2]
                    for archive in latest_archives:
                        # Füge Archive direkt zur ZIP hinzu
                        zipf.write(archive, archive.name)
                        log_message(f"📦 Hinzugefügt: {archive.name}")
                
                # Füge auch Metrik-Dateien hinzu
                metric_files = list(data_output.glob("*.json"))
                for metric_file in metric_files:
                    zipf.write(metric_file, metric_file.name)
                    log_message(f"📊 Hinzugefügt: {metric_file.name}")

            # Falls keine Archive vorhanden, sammle direkt aus Modulen
            else:
                modules_dir = Path("../pipeline/modules")
                if modules_dir.exists():
                    output_dirs = [
                        ("01_convert", "markdown_files"),
                        ("02_wiki", "wiki_json"),
                        ("03_instructQA", "qa_json"),
                        ("04_format", "formatted_dataset"),
                        ("05_bmcreator", "benchmark_questions"),
                        ("06_finetuning", "model_files"),
                        ("07_benchmark", "benchmark_results")
                    ]

                    total_files = 0
                    for module_name, zip_folder in output_dirs:
                        output_dir = modules_dir / module_name / "OUTPUT"
                        if output_dir.exists():
                            for file_path in output_dir.rglob("*"):
                                if file_path.is_file():
                                    arcname = f"{zip_folder}/{file_path.name}"
                                    zipf.write(file_path, arcname)
                                    total_files += 1

                    # CustomModel hinzufügen
                    custom_model_dir = modules_dir / "06_finetuning" / "CustomModel"
                    if custom_model_dir.exists():
                        for file_path in custom_model_dir.rglob("*"):
                            if file_path.is_file():
                                rel_path = file_path.relative_to(custom_model_dir)
                                arcname = f"custom_model/{rel_path}"
                                zipf.write(file_path, arcname)
                                total_files += 1

                    if total_files == 0:
                        return None, "❌ Keine Ergebnisse zum Download gefunden"

            log_message(f"✅ ZIP erstellt")
            return str(zip_path), f"✅ ZIP-Archiv erstellt"

    except Exception as e:
        error_msg = f"❌ Fehler beim Erstellen der ZIP: {e}"
        log_message(error_msg, "ERROR")
        return None, error_msg

# Rest der app.py bleibt gleich (UI-Definition, Event-Handler, etc.)
# ... [Der Rest des Codes bleibt unverändert] ...

# ==================== AUFRÄUMEN ====================

def get_cleanup_info():
    """Gibt Information über zu löschende Ordner zurück."""
    try:
        modules_dir = Path("../pipeline/modules")
        data_dir = Path("../pipeline/data")
        
        if not modules_dir.exists():
            return "❌ Pipeline-Verzeichnis nicht gefunden"

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
                            cleanup_info.append(f"📁 {module_name}/{folder}: {file_count} Dateien ({size_mb:.1f} MB)")
                            total_files += file_count
                            total_size += folder_size

        # CustomModel
        custom_model_dir = Path("../pipeline/CustomModel")
        if custom_model_dir.exists():
            file_count = 0
            folder_size = 0
            for file_path in custom_model_dir.rglob("*"):
                if file_path.is_file():
                    file_count += 1
                    folder_size += file_path.stat().st_size
            
            if file_count > 0:
                size_mb = folder_size / (1024 * 1024)
                cleanup_info.append(f"📁 CustomModel: {file_count} Dateien ({size_mb:.1f} MB)")
                total_files += file_count
                total_size += folder_size

        # data/OUTPUT
        if data_dir.exists():
            data_output = data_dir / "OUTPUT"
            if data_output.exists():
                file_count = 0
                folder_size = 0
                for file_path in data_output.rglob("*"):
                    if file_path.is_file():
                        file_count += 1
                        folder_size += file_path.stat().st_size

                if file_count > 0:
                    size_mb = folder_size / (1024 * 1024)
                    cleanup_info.append(f"📁 data/OUTPUT: {file_count} Dateien ({size_mb:.1f} MB)")
                    total_files += file_count
                    total_size += folder_size

        if not cleanup_info:
            return "✅ Keine Dateien zum Aufräumen gefunden"

        total_size_mb = total_size / (1024 * 1024)
        info_text = f"🗑️ Aufräum-Übersicht:\n\n"
        info_text += "\n".join(cleanup_info)
        info_text += f"\n\n📊 Gesamt: {total_files} Dateien ({total_size_mb:.1f} MB)"
        info_text += f"\n\n⚠️ WARNUNG: Diese Aktion kann nicht rückgängig gemacht werden!"

        return info_text

    except Exception as e:
        return f"❌ Fehler beim Analysieren: {e}"

def cleanup_pipeline_folders():
    """Löscht alle Arbeitsverzeichnisse."""
    try:
        log_message("🗑️ Starte Aufräumen...")
        modules_dir = Path("../pipeline/modules")
        data_dir = Path("../pipeline/data")

        if not modules_dir.exists():
            return "❌ Pipeline-Verzeichnis nicht gefunden"

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
                            log_message(f"🗑️ Gelöscht: {module_name}/{folder} ({file_count} Dateien)")

        # CustomModel aufräumen
        custom_model_dir = Path("../pipeline/CustomModel")
        if custom_model_dir.exists():
            file_count = sum(1 for f in custom_model_dir.rglob("*") if f.is_file())
            if file_count > 0:
                shutil.rmtree(custom_model_dir)
                deleted_files += file_count
                deleted_folders += 1
                log_message(f"🗑️ Gelöscht: CustomModel ({file_count} Dateien)")

        # data/OUTPUT aufräumen
        if data_dir.exists():
            data_output = data_dir / "OUTPUT"
            if data_output.exists():
                file_count = sum(1 for f in data_output.rglob("*") if f.is_file())
                if file_count > 0:
                    shutil.rmtree(data_output)
                    deleted_files += file_count
                    deleted_folders += 1
                    log_message(f"🗑️ Gelöscht: data/OUTPUT ({file_count} Dateien)")

        if deleted_files == 0:
            return "✅ Keine Dateien zum Aufräumen gefunden"

        log_message(f"✅ Aufräumen abgeschlossen: {deleted_files} Dateien in {deleted_folders} Ordnern gelöscht")
        return f"✅ Aufräumen erfolgreich!\n\n📊 Gelöscht:\n- {deleted_files} Dateien\n- {deleted_folders} Ordner\n\n🎯 Pipeline bereit für Neustart"

    except Exception as e:
        error_msg = f"❌ Fehler beim Aufräumen: {e}"
        log_message(error_msg, "ERROR")
        return error_msg

# ==================== HAUPTINTERFACE ====================

def create_main_interface():
    """Erstellt das Hauptinterface."""
    
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
        .terminal-output::-webkit-scrollbar {
            width: 8px;
        }
        .terminal-output::-webkit-scrollbar-track {
            background: #2a2a2a;
            border-radius: 4px;
        }
        .terminal-output::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 4px;
        }
        .terminal-output::-webkit-scrollbar-thumb:hover {
            background: #777;
        }
        .status-info {
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #b8d4f2;
        }
        """
    ) as interface:

        # Header ohne Logo
        gr.HTML("""
        <div class="header-gradient">
            <div style="font-size: 4em; margin-bottom: 10px;">🎯</div>
            <h1 style="margin: 0; font-size: 3em; font-weight: bold;">OpenTuneWeaver</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.3em; opacity: 0.9;">Finetuning Pipeline UI</p>
        </div>
        """)

        gr.Markdown("**Grafische Benutzeroberfläche für die komplette Finetuning-Pipeline**")

        with gr.Tabs():

            # ==================== HAUPTSEITE ====================
            with gr.TabItem("🏠 Hauptseite"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Dokument-Upload")
                        file_upload = gr.File(
                            label="Dateien auswählen (PDF, DOCX, TXT, MD, etc.)",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".md", ".html", ".xml", ".pptx", ".xlsx", ".rtf", ".odt"]
                        )
                        upload_btn = gr.Button("📤 Dateien hochladen", variant="primary")
                        upload_status = gr.Textbox(
                            label="Upload-Status",
                            interactive=False,
                            elem_classes=["status-info"]
                        )

                        gr.Markdown("### 🚀 Pipeline-Steuerung")
                        gr.Markdown("*Die Pipeline läuft vollständig automatisiert - alle Eingaben werden automatisch beantwortet*")
                        with gr.Row():
                            start_btn = gr.Button("🚀 Pipeline starten", variant="primary", size="lg")
                            stop_btn = gr.Button("⏹️ Pipeline stoppen", variant="stop")
                        
                        pipeline_status = gr.Textbox(
                            label="Pipeline-Status",
                            interactive=False,
                            elem_classes=["status-info"]
                        )

                        gr.Markdown("### 📥 Download & Aufräumen")
                        with gr.Row():
                            download_btn = gr.Button("📦 Ergebnisse herunterladen", variant="secondary")
                            cleanup_btn = gr.Button("🗑️ Aufräumen", variant="stop")
                        
                        download_status = gr.Textbox(
                            label="Download-Status",
                            interactive=False,
                            elem_classes=["status-info"]
                        )
                        
                        download_file = gr.File(
                            label="Download-Datei",
                            visible=False
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Pipeline-Übersicht")
                        gr.Markdown("""
                        **Die OpenTuneWeaver Pipeline umfasst 8 Schritte:**
                        
                        1. **📄 Document Conversion** - Konvertierung von Dokumenten zu Markdown
                        2. **📚 Wiki Generation** - Erstellung von Wiki-Strukturen  
                        3. **❓ Instruct QA Creation** - Generierung von Frage-Antwort-Paaren
                        4. **🔧 Dataset Formatting** - Formatierung für das Training
                        5. **📊 Benchmark Creation** - Erstellung von Benchmark-Fragen
                        6. **🤖 Fine-tuning** - Training des Modells
                        7. **🏁 Benchmarking** - Evaluation des trainierten Modells
                        8. **📦 Results Archive** - Archivierung aller Ergebnisse
                        
                        **Verwendung:**
                        1. Laden Sie Ihre Dokumente hoch
                        2. Konfigurieren Sie die Einstellungen (siehe Einstellungen-Tab)
                        3. Starten Sie die Pipeline (läuft vollautomatisch)
                        4. Verfolgen Sie den Fortschritt im Terminal-Tab
                        5. Laden Sie die Ergebnisse herunter
                        
                        **🤖 Automatisierung:** Die Pipeline beantwortet alle Eingabeaufforderungen automatisch:
                        - Option 1 (Komplette Pipeline)
                        - Keine neue Konfiguration (verwendet UI-Einstellungen)
                        - Standard-Werte für alle Parameter
                        - Automatische Bereinigung am Ende
                        
                        **💡 Hinweis:** Wenn Sie das UI-Fenster schließen, wird die Pipeline unterbrochen.
                        """)

            # ==================== EINSTELLUNGEN ====================
            with gr.TabItem("⚙️ Einstellungen"):
                gr.Markdown("### 🔧 Pipeline-Konfiguration")
                gr.Markdown("Hier können Sie alle Einstellungen der Pipeline konfigurieren.")

                with gr.Accordion("🔑 HuggingFace Tokens", open=True):
                    with gr.Row():
                        hf_token = gr.Textbox(
                            label="HF Token (für Modell-Downloads)",
                            type="password",
                            placeholder="hf_..."
                        )
                        hf_write_token = gr.Textbox(
                            label="HF Write Token (optional)",
                            type="password",
                            placeholder="hf_..."
                        )

                with gr.Accordion("🌐 API-Konfiguration", open=True):
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

                    gr.Markdown("**Modell-Konfiguration für jeden Schritt:**")
                    with gr.Row():
                        convert_model = gr.Textbox(
                            label="📄 Convert Modell",
                            value="gemma3:12b-it-qat"
                        )
                        convert_temp = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            value=0.1,
                            step=0.1
                        )

                    with gr.Row():
                        wiki_model = gr.Textbox(
                            label="📚 Wiki Modell",
                            value="gemma3:12b-it-qat"
                        )
                        wiki_temp = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            value=0.3,
                            step=0.1
                        )

                    with gr.Row():
                        qa_model = gr.Textbox(
                            label="❓ QA Modell",
                            value="gemma3:12b-it-qat"
                        )
                        qa_temp = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1
                        )

                    with gr.Row():
                        benchmark_model = gr.Textbox(
                            label="📊 Benchmark Modell",
                            value="gemma3:12b-it-qat"
                        )
                        benchmark_temp = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            value=0.5,
                            step=0.1
                        )

                with gr.Accordion("🤖 Fine-tuning Konfiguration", open=True):
                    with gr.Row():
                        model_name = gr.Textbox(
                            label="Modell-Name",
                            value="OpenTuneWeaver-Model"
                        )
                        base_model = gr.Textbox(
                            label="Base Model",
                            value="unsloth/gemma-3n-E2B-it"
                        )

                    with gr.Row():
                        hf_repo_id = gr.Textbox(
                            label="HuggingFace Repo ID",
                            value="user/OpenTuneWeaver-Model"
                        )
                        custom_model_dir = gr.Textbox(
                            label="CustomModel Verzeichnis",
                            value="CustomModel"
                        )

                    gr.Markdown("**Training-Parameter:**")
                    with gr.Row():
                        max_seq_length = gr.Slider(
                            label="Max Sequence Length",
                            minimum=512,
                            maximum=16384,
                            value=8192,
                            step=512
                        )
                        load_in_4bit = gr.Checkbox(
                            label="Load in 4-bit",
                            value=True
                        )
                        full_finetuning = gr.Checkbox(
                            label="Full Fine-tuning",
                            value=False
                        )

                    gr.Markdown("**LoRA-Parameter:**")
                    with gr.Row():
                        lora_r = gr.Slider(
                            label="LoRA r",
                            minimum=1,
                            maximum=64,
                            value=8,
                            step=1
                        )
                        lora_alpha = gr.Slider(
                            label="LoRA Alpha",
                            minimum=1,
                            maximum=64,
                            value=8,
                            step=1
                        )
                        lora_dropout = gr.Slider(
                            label="LoRA Dropout",
                            minimum=0.0,
                            maximum=0.5,
                            value=0.0,
                            step=0.05
                        )

                    gr.Markdown("**Training-Einstellungen:**")
                    with gr.Row():
                        batch_size = gr.Slider(
                            label="Batch Size",
                            minimum=1,
                            maximum=16,
                            value=1,
                            step=1
                        )
                        grad_accumulation = gr.Slider(
                            label="Gradient Accumulation Steps",
                            minimum=1,
                            maximum=64,
                            value=16,
                            step=1
                        )

                    with gr.Row():
                        warmup_steps = gr.Slider(
                            label="Warmup Steps",
                            minimum=0,
                            maximum=1000,
                            value=200,
                            step=10
                        )
                        num_epochs = gr.Slider(
                            label="Anzahl Epochen",
                            minimum=1,
                            maximum=20,
                            value=3,
                            step=1
                        )

                    with gr.Row():
                        learning_rate = gr.Slider(
                            label="Learning Rate",
                            minimum=1e-6,
                            maximum=1e-3,
                            value=5e-5,
                            step=1e-6
                        )
                        weight_decay = gr.Slider(
                            label="Weight Decay",
                            minimum=0.0,
                            maximum=0.3,
                            value=0.03,
                            step=0.01
                        )

                    gr.Markdown("**Output-Optionen:**")
                    with gr.Row():
                        save_lora = gr.Checkbox(
                            label="LoRA Adapter speichern",
                            value=True
                        )
                        save_merged = gr.Checkbox(
                            label="Merged Model speichern",
                            value=True
                        )
                        save_gguf = gr.Checkbox(
                            label="GGUF Model speichern",
                            value=False
                        )

                with gr.Accordion("🏁 Benchmark-Konfiguration", open=False):
                    with gr.Row():
                        benchmark_mode = gr.Dropdown(
                            label="Benchmark Modus",
                            choices=["comparison", "post_only", "pre_only"],
                            value="comparison"
                        )
                        evaluator_model = gr.Textbox(
                            label="Evaluator Model",
                            value="gemma3:12b-it-qat"
                        )

                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            label="Max New Tokens",
                            minimum=50,
                            maximum=1000,
                            value=256,
                            step=10
                        )
                        eval_temp = gr.Slider(
                            label="Evaluation Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            value=0.3,
                            step=0.1
                        )

                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top P",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05
                        )
                        top_k = gr.Slider(
                            label="Top K",
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1
                        )

                    with gr.Row():
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            minimum=1.0,
                            maximum=2.0,
                            value=1.1,
                            step=0.05
                        )

                with gr.Accordion("⚙️ Pipeline-Einstellungen", open=False):
                    with gr.Row():
                        auto_cleanup = gr.Checkbox(
                            label="Automatisches Aufräumen",
                            value=False
                        )
                        verbose_mode = gr.Checkbox(
                            label="Verbose Modus",
                            value=True
                        )
                        continue_on_error = gr.Checkbox(
                            label="Bei Fehlern fortfahren",
                            value=True
                        )

                # Buttons
                with gr.Row():
                    load_config_btn = gr.Button("📋 Aktuelle Konfiguration laden", variant="secondary")
                    save_config_btn = gr.Button("💾 Konfiguration speichern", variant="primary")

                config_status = gr.Textbox(
                    label="Konfigurations-Status",
                    interactive=False,
                    elem_classes=["status-info"]
                )

            # ==================== TERMINAL ====================
            with gr.TabItem("🖥️ Terminal"):
                gr.Markdown("### 🖥️ Live-Terminal Ausgabe")
                gr.Markdown("*Das Terminal zeigt alle Pipeline-Schritte und automatischen Antworten in Echtzeit*")
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Aktualisieren", variant="secondary")
                    clear_btn = gr.Button("🗑️ Löschen", variant="secondary")

                terminal_output = gr.Textbox(
                    label="Terminal-Ausgabe",
                    value="📋 OpenTuneWeaver Pipeline bereit...\n💡 Laden Sie Dateien hoch, konfigurieren Sie die Einstellungen und starten Sie die Pipeline\n🤖 Die Pipeline läuft vollautomatisch - alle Eingaben werden automatisch beantwortet",
                    interactive=False,
                    lines=30,
                    max_lines=30,
                    elem_classes=["terminal-output"]
                )

            # ==================== AUFRÄUMEN ====================
            with gr.TabItem("🗑️ Aufräumen"):
                with gr.Column():
                    gr.Markdown("### 🧹 Pipeline-Verzeichnisse aufräumen")
                    gr.Markdown("Löschen Sie alle temporären Dateien und Ergebnisse für einen sauberen Neustart")

                    with gr.Row():
                        with gr.Column(scale=1):
                            analyze_btn = gr.Button("🔍 Verzeichnisse analysieren", variant="secondary")
                            cleanup_confirm_btn = gr.Button("🗑️ ALLES LÖSCHEN", variant="stop")

                            cleanup_result = gr.Textbox(
                                label="Aufräum-Ergebnis",
                                interactive=False,
                                elem_classes=["status-info"]
                            )

                        with gr.Column(scale=2):
                            cleanup_info = gr.Textbox(
                                label="Aufräum-Analyse",
                                value="📊 Klicken Sie auf 'Verzeichnisse analysieren' um zu sehen was gelöscht werden würde",
                                interactive=False,
                                lines=20
                            )

                    gr.Markdown("### ⚠️ Wichtige Hinweise")
                    gr.Markdown("""
                    **Was wird gelöscht:**
                    - 📁 Alle INPUT/OUTPUT Verzeichnisse der Module
                    - 📁 UPLOAD Verzeichnis mit hochgeladenen Dateien
                    - 📁 CustomModel Verzeichnis mit trainierten Modellen
                    - 📁 data/OUTPUT Verzeichnis mit archivierten Ergebnissen
                    - 📁 Alle Benchmark-Verzeichnisse

                    **Was bleibt erhalten:**
                    - ✅ Alle Python-Skripte der Pipeline
                    - ✅ Konfigurationsdateien
                    - ✅ Diese UI-Anwendung
                    - ✅ Viewer-Dateien

                    **⚠️ Diese Aktion kann NICHT rückgängig gemacht werden!**
                    **Erstellen Sie vorher ein Backup oder laden Sie die Ergebnisse herunter!**
                    """)

        # ==================== EVENT-HANDLER ====================

        # Upload-Handler
        upload_btn.click(
            fn=save_uploaded_files,
            inputs=[file_upload],
            outputs=[upload_status]
        )

        # Pipeline-Handler
        start_btn.click(
            fn=start_pipeline,
            outputs=[pipeline_status]
        )

        stop_btn.click(
            fn=stop_pipeline,
            outputs=[pipeline_status]
        )

        # Terminal-Handler
        refresh_btn.click(
            fn=get_terminal_output,
            outputs=[terminal_output]
        )

        clear_btn.click(
            fn=clear_terminal,
            outputs=[terminal_output]
        )

        # Download-Handler
        def handle_download():
            zip_path, status = create_download_zip()
            if zip_path:
                return status, zip_path, gr.update(visible=True)
            else:
                return status, None, gr.update(visible=False)

        download_btn.click(
            fn=handle_download,
            outputs=[download_status, download_file, download_file]
        )

        # Aufräum-Handler
        analyze_btn.click(
            fn=get_cleanup_info,
            outputs=[cleanup_info]
        )

        cleanup_btn.click(
            fn=cleanup_pipeline_folders,
            outputs=[cleanup_result]
        )

        cleanup_confirm_btn.click(
            fn=cleanup_pipeline_folders,
            outputs=[cleanup_result]
        )

        # Konfigurations-Handler
        def load_config_to_ui():
            config = load_existing_config()
            
            # Tokens
            tokens = config.get("tokens", {})
            hf_token_val = tokens.get("hf_token", "")
            hf_write_token_val = tokens.get("hf_write_token", "")
            
            # API Config (nehme ersten Eintrag als Referenz)
            api_config = config.get("api_configs", {}).get("01_convert", {})
            api_base_url_val = api_config.get("openai_base_url", "http://localhost:11434/v1")
            api_key_val = api_config.get("openai_api_key", "ollama")
            
            # Modelle
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
                "✅ Konfiguration geladen"
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

        # Auto-Refresh für Terminal
        def auto_scroll_terminal():
            output = get_terminal_output()
            return output

        try:
            interface.load(
                fn=auto_scroll_terminal,
                outputs=[terminal_output],
                every=2
            )
            log_message("✅ Auto-Refresh aktiviert (2 Sekunden)")
        except:
            log_message("⚠️ Auto-Refresh nicht unterstützt - verwenden Sie 'Aktualisieren'")

        return interface

def main():
    """Hauptfunktion."""
    print("="*80)
    print(" 🎯 OPENTUNEWEAVER - FINETUNING PIPELINE UI")
    print("="*80)
    log_message("🚀 Starte OpenTuneWeaver UI...")
    log_message("✨ Features: Upload, Konfiguration, Pipeline, Terminal, Download, Aufräumen")

    # Erstelle Interface
    interface = create_main_interface()

    # Starte Server
    log_message("🌐 Starte Web-Server...")
    print("\n🌐 UI verfügbar unter:")
    print(" - Lokal: http://localhost:8080")
    print(" - Netzwerk: http://YOUR_IP:8080")
    print("\n🎯 Features:")
    print(" - 📤 Datei-Upload für alle Dokumentformate")
    print(" - ⚙️ Vollständige Pipeline-Konfiguration")
    print(" - 🖥️ Live-Terminal mit Auto-Refresh und Auto-Scroll")
    print(" - 📦 Download aller Ergebnisse als ZIP")
    print(" - 🤖 Vollautomatische Pipeline-Ausführung")
    print(" - 🗑️ Umfassendes Aufräumen")

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
        print(f"\n❌ Server-Fehler: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ OpenTuneWeaver UI beendet")
    except Exception as e:
        print(f"\n💥 Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
