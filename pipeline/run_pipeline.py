#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

OpenTuneWeaver Pipeline Runner - SIMPLIFIED with central configuration

Version: 5.3 - With archiving functionality, viewer integration and metrics collection

"""

import os
import sys
import json
import time
import subprocess
import shutil
import getpass
import zipfile
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from glob import glob

# Pipeline steps definition
STEPS = [
    {
        'id': 1,
        'name': 'Document Conversion',
        'script': 'modules/01_convert/01_convert.py',
        'working_dir': 'modules/01_convert',
        'directories': ['UPLOAD', 'INPUT', 'OUTPUT'],
        'needs_api': True
    },
    {
        'id': 2,
        'name': 'Wiki Generation',
        'script': 'modules/02_wiki/02_genwiki.py',
        'working_dir': 'modules/02_wiki',
        'directories': ['INPUT', 'OUTPUT'],
        'needs_api': True
    },
    {
        'id': 3,
        'name': 'Instruct QA Creation',
        'script': 'modules/03_instructQA/03_instructQA.py',
        'working_dir': 'modules/03_instructQA',
        'directories': ['INPUT', 'OUTPUT'],
        'needs_api': True
    },
    {
        'id': 4,
        'name': 'Dataset Formatting',
        'script': 'modules/04_format/04_format.py',
        'working_dir': 'modules/04_format',
        'directories': ['INPUT', 'OUTPUT']
    },
    {
        'id': 5,
        'name': 'Benchmark Creation',
        'script': 'modules/05_bmcreator/05_bmcreator.py',
        'working_dir': 'modules/05_bmcreator',
        'directories': ['INPUT', 'BENCHMARKFRAGEN'],
        'needs_api': True
    },
    {
        'id': 6,
        'name': 'Fine-tuning',
        'script': 'modules/06_finetuning/06_finetuning.py',
        'working_dir': 'modules/06_finetuning',
        'directories': ['INPUT', 'CustomModel'],
        'needs_hf_token': True
    },
    {
        'id': 7,
        'name': 'Benchmarking',
        'script': 'modules/07_benchmark/07_benchmark.py',
        'working_dir': 'modules/07_benchmark',
        'directories': ['BENCHMARKFRAGEN', 'OUTPUT']
    },
    {
        'id': 8,
        'name': 'Results Archive & Transfer',
        'script': None,  # Executed directly
        'working_dir': '.',
        'directories': ['data/OUTPUT']
    }
]

class SimplifiedPipelineRunner:
    """Simplified Pipeline Runner with central configuration and metrics collection"""
    def write_status_update(self, step_id, status, stats=None, duration=None):
        '''Writes status update to JSON file for UI'''
        status_file = Path('pipeline_status.json')
        
        # Read existing status or create new
        if status_file.exists():
            with open(status_file, 'r') as f:
                current_status = json.load(f)
        else:
            current_status = {}
        
        # Update status (use string keys for JSON compatibility)
        step_key = str(step_id)
        if step_key not in current_status:
            current_status[step_key] = {
                'name': STEPS[step_id-1]['name'],
                'icon': ['📄','📚','❓','🔧','📊','🤖','🏆','📦'][step_id-1],
                'status': 'pending',
                'stats': {},
                'duration': None
            }
        
        current_status[step_key]['status'] = status
        if stats:
            current_status[step_key]['stats'] = stats
        if duration:
            current_status[step_key]['duration'] = duration
        
        # Write back
        with open(status_file, 'w') as f:
            json.dump(current_status, f)

    def __init__(self):
        self.start_time = datetime.now()
        self.config_file = "pipeline_config.json"
        self.config = {}
        self.auto_mode = False  # Flag for automated mode
        self.use_existing_config = True  # Flag for config usage
        self.cleanup_after = False  # Flag for cleanup after pipeline
        self.metrics = {
            'pipeline_version': '5.3',
            'start_time': self.start_time.isoformat(),
            'end_time': None,
            'total_duration': None,
            'model_name': None,
            'steps': {},
            'data_statistics': {},
            'benchmark_results': {},
            'system_info': self.get_system_info(),
            'errors': [],
            'warnings': []
        }

    def get_system_info(self):
        """Captures system information"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                gpu_name = "N/A"
                gpu_memory = 0
        except:
            cuda_available = False
            gpu_name = "N/A"
            gpu_memory = 0
        
        return {
            'python_version': sys.version.split()[0],
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'cuda_available': cuda_available,
            'gpu_name': gpu_name,
            'gpu_memory_gb': round(gpu_memory, 1)
        }

    def analyze_lexikon_files(self, output_dir):
        """Analyzes lexicon JSON files"""
        stats = {
            'total_entries': 0,
            'categories': {},
            'file_count': 0,
            'total_size_mb': 0
        }
        
        lexikon_files = glob(os.path.join(output_dir, 'lexikon_*.json'))
        stats['file_count'] = len(lexikon_files)
        
        for file_path in lexikon_files:
            try:
                stats['total_size_mb'] += os.path.getsize(file_path) / (1024*1024)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'lexikon' in data:
                    for entry in data['lexikon']:
                        stats['total_entries'] += 1
                        category = entry.get('kategorie', 'Unknown')
                        stats['categories'][category] = stats['categories'].get(category, 0) + 1
            except Exception as e:
                self.metrics['warnings'].append(f"Error analyzing {file_path}: {str(e)}")
        
        return stats

    def analyze_qa_dataset(self, output_dir):
        """Analyzes QA/Instruct datasets"""
        stats = {
            'total_qa_pairs': 0,
            'instruction_types': {},
            'file_count': 0,
            'total_size_mb': 0,
            'avg_question_length': 0,
            'avg_answer_length': 0
        }
        
        qa_files = glob(os.path.join(output_dir, '*.json'))
        stats['file_count'] = len(qa_files)
        
        question_lengths = []
        answer_lengths = []
        
        for file_path in qa_files:
            try:
                stats['total_size_mb'] += os.path.getsize(file_path) / (1024*1024)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Support different formats
                if isinstance(data, list):
                    entries = data
                elif 'qa_pairs' in data:
                    entries = data['qa_pairs']
                elif 'dataset' in data:
                    entries = data['dataset']
                else:
                    entries = []
                
                for entry in entries:
                    stats['total_qa_pairs'] += 1
                    
                    # Analyze instruction types
                    if 'instruction' in entry:
                        inst_type = entry.get('type', 'standard')
                        stats['instruction_types'][inst_type] = stats['instruction_types'].get(inst_type, 0) + 1
                    
                    # Length statistics
                    if 'question' in entry:
                        question_lengths.append(len(entry['question']))
                    if 'answer' in entry:
                        answer_lengths.append(len(entry['answer']))
                    elif 'response' in entry:
                        answer_lengths.append(len(entry['response']))
                        
            except Exception as e:
                self.metrics['warnings'].append(f"Error analyzing {file_path}: {str(e)}")
        
        if question_lengths:
            stats['avg_question_length'] = round(sum(question_lengths) / len(question_lengths))
        if answer_lengths:
            stats['avg_answer_length'] = round(sum(answer_lengths) / len(answer_lengths))
        
        return stats

    def analyze_benchmark_questions(self, benchmark_dir):
        """Analyzes benchmark questions"""
        stats = {
            'total_questions': 0,
            'categories': {},
            'difficulty_distribution': {},
            'file_size_mb': 0
        }
        
        benchmark_file = os.path.join(benchmark_dir, 'benchmark_fragen_complete.json')
        
        if os.path.exists(benchmark_file):
            try:
                stats['file_size_mb'] = os.path.getsize(benchmark_file) / (1024*1024)
                
                with open(benchmark_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'kategorien' in data:
                    for kategorie_data in data['kategorien']:
                        kategorie = kategorie_data.get('kategorie', 'Unknown')
                        fragen = kategorie_data.get('fragen', [])
                        stats['categories'][kategorie] = len(fragen)
                        stats['total_questions'] += len(fragen)
                        
                        # Difficulty distribution (if available)
                        for frage in fragen:
                            difficulty = frage.get('schwierigkeit', 'medium')
                            stats['difficulty_distribution'][difficulty] = stats['difficulty_distribution'].get(difficulty, 0) + 1
                            
            except Exception as e:
                self.metrics['warnings'].append(f"Error analyzing benchmark questions: {str(e)}")
        
        return stats

    def analyze_benchmark_results(self, output_dir):
        """Analyzes benchmark results"""
        results = {
            'comparison_available': False,
            'pre_score': None,
            'post_score': None,
            'improvement': None,
            'category_scores': {},
            'detailed_results': {}
        }
        
        # Search for latest benchmark results
        result_files = glob(os.path.join(output_dir, 'comparison_*.json')) + \
                      glob(os.path.join(output_dir, 'post_only_*.json')) + \
                      glob(os.path.join(output_dir, 'pre_only_*.json'))
        
        if result_files:
            # Take the newest file
            latest_file = max(result_files, key=os.path.getctime)
            
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract scores
                if 'pre' in data:
                    results['pre_score'] = data['pre'].get('percentage_score')
                if 'post' in data:
                    results['post_score'] = data['post'].get('percentage_score')
                
                if results['pre_score'] and results['post_score']:
                    results['comparison_available'] = True
                    results['improvement'] = results['post_score'] - results['pre_score']
                
                # Analyze category scores
                for model_type in ['pre', 'post']:
                    if model_type in data and 'results' in data[model_type]:
                        category_scores = {}
                        for result in data[model_type]['results']:
                            category = result.get('category', 'Unknown')
                            score = result.get('score', 0)
                            if category not in category_scores:
                                category_scores[category] = []
                            category_scores[category].append(score)
                        
                        # Calculate average per category
                        for category, scores in category_scores.items():
                            avg_score = sum(scores) / len(scores) if scores else 0
                            if model_type not in results['category_scores']:
                                results['category_scores'][model_type] = {}
                            results['category_scores'][model_type][category] = round(avg_score, 2)
                
                results['benchmark_file'] = os.path.basename(latest_file)
                results['benchmark_timestamp'] = datetime.fromtimestamp(os.path.getctime(latest_file)).isoformat()
                
            except Exception as e:
                self.metrics['warnings'].append(f"Error analyzing benchmark results: {str(e)}")
        
        return results

    def analyze_training_results(self, finetuning_dir):
        """Analyzes training results"""
        stats = {
            'model_trained': False,
            'training_duration': None,
            'final_loss': None,
            'model_size_gb': 0,
            'adapter_size_mb': 0,
            'merged_model_available': False,
            'gguf_available': False
        }
        
        # Check latest_model_info.json
        info_file = os.path.join(finetuning_dir, 'latest_model_info.json')
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                stats['model_trained'] = True
                stats['training_duration'] = info.get('training_time')
                stats['final_loss'] = info.get('final_loss')
            except:
                pass
        
        # Analyze CustomModel directory
        custom_model_dir = os.path.join(finetuning_dir, 'CustomModel')
        if os.path.exists(custom_model_dir):
            model_name = self.config.get('finetuning', {}).get('model_name', 'model')
            
            # Adapter
            adapter_dir = os.path.join(custom_model_dir, model_name)
            if os.path.exists(adapter_dir):
                for root, dirs, files in os.walk(adapter_dir):
                    for file in files:
                        if file.endswith(('.bin', '.safetensors')):
                            stats['adapter_size_mb'] += os.path.getsize(os.path.join(root, file)) / (1024*1024)
            
            # Merged Model
            merged_dir = os.path.join(custom_model_dir, f"{model_name}_merged")
            if os.path.exists(merged_dir):
                stats['merged_model_available'] = True
                for root, dirs, files in os.walk(merged_dir):
                    for file in files:
                        if file.endswith(('.bin', '.safetensors')):
                            stats['model_size_gb'] += os.path.getsize(os.path.join(root, file)) / (1024**3)
            
            # GGUF
            gguf_files = glob(os.path.join(custom_model_dir, '*.gguf'))
            if gguf_files:
                stats['gguf_available'] = True
                stats['gguf_files'] = [os.path.basename(f) for f in gguf_files]
        
        return stats

    def detect_post_model_path(self, custom_model_dir, model_name):
        """
        Intelligent detection of post-model path
        Checks various possibilities and returns the best path
        """
        print("\n🔍 Searching for trained model...")
        
        # Check possible paths
        candidates = [
            # 1. Merged model (fully merged)
            {
                'path': f"{custom_model_dir}/{model_name}_merged",
                'type': 'merged',
                'description': 'Merged Model'
            },
            # 2. Adapter (LoRA)
            {
                'path': f"{custom_model_dir}/{model_name}",
                'type': 'adapter',
                'description': 'LoRA Adapter'
            },
            # 3. Alternative paths
            {
                'path': f"./modules/06_finetuning/{custom_model_dir}/{model_name}_merged",
                'type': 'merged',
                'description': 'Merged (Module Path)'
            },
            {
                'path': f"./modules/06_finetuning/{custom_model_dir}/{model_name}",
                'type': 'adapter',
                'description': 'Adapter (Module Path)'
            }
        ]
        
        found_models = []
        for candidate in candidates:
            path = candidate['path']
            if os.path.exists(path):
                # Check if it's a valid model/adapter
                is_valid = False
                if candidate['type'] == 'merged':
                    # Check for model files
                    required_files = ['config.json', 'tokenizer_config.json']
                    model_files = ['model.safetensors', 'pytorch_model.bin', 'model-00001-of-*.safetensors']
                    has_config = any(os.path.exists(os.path.join(path, f)) for f in required_files)
                    has_model = any(
                        glob(os.path.join(path, pattern))
                        for pattern in model_files
                    )
                    is_valid = has_config or has_model
                    if is_valid:
                        print(f" ✅ Found: {candidate['description']} in {path}")
                        found_models.append(candidate)
                    else:
                        print(f" ⚠️ Directory exists but incomplete: {path}")
                elif candidate['type'] == 'adapter':
                    # Check for adapter files
                    adapter_files = ['adapter_config.json', 'adapter_model.safetensors', 'adapter_model.bin']
                    has_adapter = any(os.path.exists(os.path.join(path, f)) for f in adapter_files)
                    if has_adapter:
                        print(f" ✅ Found: {candidate['description']} in {path}")
                        found_models.append(candidate)
                        is_valid = True
                    else:
                        print(f" ⚠️ Directory exists but no adapter: {path}")
        
        # Decide which model to use
        if found_models:
            # Preference: 1. Merged, 2. Adapter
            merged_models = [m for m in found_models if m['type'] == 'merged']
            adapter_models = [m for m in found_models if m['type'] == 'adapter']
            
            if merged_models:
                chosen = merged_models[0]
                print(f"\n📦 Using: {chosen['description']}")
                return chosen['path'], 'merged'
            elif adapter_models:
                chosen = adapter_models[0]
                print(f"\n🔧 Using: {chosen['description']}")
                print(" ℹ️ Note: Benchmark script will load Base+Adapter")
                return chosen['path'], 'adapter'
        
        # Fallback
        print("\n⚠️ No trained model found!")
        print(" Using fallback path for later detection")
        return f"{custom_model_dir}/{model_name}", 'unknown'

    def create_default_config(self):
        """Creates a default configuration without user input (for auto mode)"""
        print("\n" + "="*60)
        print("📋 CREATING DEFAULT CONFIGURATION (AUTO MODE)")
        print("="*60)
        
        config = {
            "version": "5.3",
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "auto_generated": True,
            "tokens": {
                "hf_token": os.environ.get("HF_TOKEN", ""),
                "hf_write_token": os.environ.get("HF_WRITE_TOKEN", "")
            },
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
                "model_name": "AutoModel",
                "base_model": "unsloth/gemma-3n-E2B-it",
                "hf_repo_id": "user/AutoModel",
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
                    "name": "CustomModel/AutoModel",
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
                "save_metrics": True,
                "capture_output_stats": True
            }
        }
        
        # Save
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Default configuration saved: {self.config_file}")
        self.config = config
        self.metrics['model_name'] = "AutoModel"
        return config

    def create_central_config(self):
        """Creates the central configuration file"""
        print("\n" + "="*60)
        print("📋 CENTRAL PIPELINE CONFIGURATION")
        print("="*60)
        
        config = {
            "version": "5.3",
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        
        # 1. HuggingFace Tokens
        print("\n🔑 HUGGINGFACE TOKENS:")
        print("Required for fine-tuning (Step 6)")
        setup_tokens = input("Configure HF tokens? (y/n) [y]: ").lower().strip() != 'n'
        
        if setup_tokens:
            print("\n1. READ Token (for model downloads):")
            hf_token = getpass.getpass("Enter HF_TOKEN (hidden): ").strip()
            if not hf_token:
                hf_token = input("Use dummy token? (y/n) [n]: ").lower() == 'y'
                if hf_token:
                    hf_token = "hf_DUMMY_TOKEN_REPLACE_ME"
                    print("⚠️ Dummy token set - replace later!")
            
            print("\n2. WRITE Token (for uploads, optional):")
            hf_write = input("Set write token? (y/n) [n]: ").lower() == 'y'
            hf_write_token = None
            if hf_write:
                hf_write_token = getpass.getpass("Enter HF_WRITE_TOKEN: ").strip()
                if not hf_write_token:
                    hf_write_token = hf_token # Fallback to READ token
            
            config["tokens"] = {
                "hf_token": hf_token,
                "hf_write_token": hf_write_token or hf_token
            }
        else:
            config["tokens"] = {
                "hf_token": "",
                "hf_write_token": ""
            }
        
        # 2. API configurations for modules
        print("\n🌐 API CONFIGURATIONS:")
        print("For modules 01-03 and 05 (data processing)")
        
        # Standard API config
        use_openai = input("Use OpenAI-compatible API? (y/n) [y]: ").lower().strip() != 'n'
        
        if use_openai:
            base_url = input("API Base URL [http://localhost:11434/v1]: ").strip()
            base_url = base_url or "http://localhost:11434/v1"
            api_key = input("API Key [ollama]: ").strip() or "ollama"
            
            print("\nModel configuration for each module:")
            
            # Module-specific models
            api_configs = {}
            
            # 01_convert
            print("\n📄 Module 01 - Document Conversion:")
            model_01 = input("Model [gemma3:12b-it-qat]: ").strip() or "gemma3:12b-it-qat"
            api_configs["01_convert"] = {
                "use_openai_api": True,
                "openai_base_url": base_url,
                "openai_api_key": api_key,
                "openai_model_name": model_01,
                "temperature": 0.1
            }
            
            # 02_genwiki
            print("\n📚 Module 02 - Wiki Generation:")
            model_02 = input("Model [gemma3:12b-it-qat]: ").strip() or "gemma3:12b-it-qat"
            api_configs["02_genwiki"] = {
                "use_openai_api": True,
                "openai_base_url": base_url,
                "openai_api_key": api_key,
                "openai_model_name": model_02,
                "temperature": 0.3
            }
            
            # 03_instructQA
            print("\n❓ Module 03 - Instruct QA:")
            model_03 = input("Model [gemma3:12b-it-qat]: ").strip() or "gemma3:12b-it-qat"
            api_configs["03_instructQA"] = {
                "use_openai_api": True,
                "openai_base_url": base_url,
                "openai_api_key": api_key,
                "openai_model_name": model_03,
                "temperature": 0.7
            }
            
            # 05_bmcreator
            print("\n📊 Module 05 - Benchmark Creation:")
            model_05 = input("Model [gemma3:12b-it-qat]: ").strip() or "gemma3:12b-it-qat"
            api_configs["05_bmcreator"] = {
                "use_openai_api": True,
                "openai_base_url": base_url,
                "openai_api_key": api_key,
                "openai_model_name": model_05,
                "temperature": 0.5
            }
            
            config["api_configs"] = api_configs
            
        else:
            # Ollama configuration
            print("\n🦙 Ollama Configuration:")
            ollama_url = input("Ollama Server URL [http://localhost:11434]: ").strip()
            ollama_url = ollama_url or "http://localhost:11434"
            
            api_configs = {}
            for module_id in ["01_convert", "02_genwiki", "03_instructQA", "05_bmcreator"]:
                model_name = input(f"Model for {module_id} [gemma3:12b-it-qat]: ").strip() or "gemma3:12b-it-qat"
                api_configs[module_id] = {
                    "use_openai_api": False,
                    "ollama_server_url": ollama_url,
                    "ollama_model_name": model_name
                }
            
            config["api_configs"] = api_configs
        
        # 3. Fine-tuning configuration
        print("\n🤖 FINE-TUNING CONFIGURATION:")
        model_name = input("Output model name [FoxLM-e2b]: ").strip() or "FoxLM-e2b"
        custom_model_dir = input("CustomModel directory [CustomModel]: ").strip() or "CustomModel"

        # New queries: save_merged and save_gguf
        print("\n📦 Model Export Options:")
        save_merged = input("Save merged model? (y/n) [y]: ").strip().lower() or 'y'
        save_gguf = input("Save GGUF model? (y/n) [n]: ").strip().lower() or 'n'
        
        config["finetuning"] = {
            "model_name": model_name,
            "base_model": "unsloth/gemma-3n-E2B-it",
            "hf_repo_id": "",
            "dataset_path": "INPUT/dataset.json",
            "chat_template": "gemma-3",
            "custom_model_dir": custom_model_dir,
            # Training settings
            "max_seq_length": 8192,
            "load_in_4bit": True,
            "full_finetuning": False,
            # LoRA settings
            "lora_r": 8,
            "lora_alpha": 8,
            "lora_dropout": 0,
            "bias": "none",
            "random_state": 3407,
            # Training parameters
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
            # Output settings
            "save_lora": True,
            "save_merged": save_merged == 'y',
            "save_gguf": save_gguf == 'y',
            "upload_to_hf": False,
            "gguf_quantizations": ["q8_0"],
            # Inference settings
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_new_tokens": 128
        }
        
        # Generate HF Repo ID
        username = input("HuggingFace Username [user]: ").strip() or "user"
        config["finetuning"]["hf_repo_id"] = f"{username}/{model_name}"
        
        # Training profile
        print("\n📊 Training Profile:")
        print("1. Test (fast)")
        print("2. Development")
        print("3. Production")
        profile = input("Profile (1-3) [2]: ").strip() or "2"
        
        if profile == "1":
            config["finetuning"]["num_train_epochs"] = 1
            config["finetuning"]["warmup_steps"] = 10
        elif profile == "3":
            config["finetuning"]["num_train_epochs"] = 10
            config["finetuning"]["warmup_steps"] = 500
        
        # 4. Benchmark configuration with intelligent post-model detection
        print("\n🏁 BENCHMARK CONFIGURATION:")
        
        # Try to find the post-model
        post_model_path, model_type = self.detect_post_model_path(custom_model_dir, model_name)
        
        config["benchmark"] = {
            "mode": "comparison", # comparison, post_only, pre_only
            "pre_model": {
                "name": "unsloth/gemma-3n-E2B-it",
                "type": "transformers",
                "load_in_4bit": False,
                "max_seq_length": 2048
            },
            "post_model": {
                "name": post_model_path, # Concrete path instead of "auto-detect"
                "type": model_type, # "merged", "adapter" or "unknown"
                "load_in_4bit": False,
                "max_seq_length": 2048,
                "base_model": "unsloth/gemma-3n-E2B-it" if model_type == "adapter" else None
            },
            "evaluator": {
                "type": "api", # api or local
                "api_base_url": base_url if use_openai else "http://localhost:11434/v1",
                "api_key": api_key if use_openai else "ollama",
                "model": "gemma3:12b-it-qat"
            },
            "questions_file": "BENCHMARKFRAGEN/benchmark_fragen_complete.json",
            "max_new_tokens": 256,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1
        }
        
        # 5. Pipeline settings
        config["pipeline"] = {
            "auto_cleanup": False,
            "verbose": True,
            "continue_on_error": True,
            "save_metrics": True,
            "capture_output_stats": True
        }
        
        # Save
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Central configuration saved: {self.config_file}")
        self.config = config
        self.metrics['model_name'] = model_name  # Save in metrics
        return config

    def update_post_model_path(self):
        """
        Updates the post-model path in an existing config
        Useful after fine-tuning
        """
        if not self.config:
            return
        
        ft_config = self.config.get("finetuning", {})
        model_name = ft_config.get("model_name", "model")
        custom_model_dir = ft_config.get("custom_model_dir", "CustomModel")
        
        post_model_path, model_type = self.detect_post_model_path(custom_model_dir, model_name)
        
        if "benchmark" in self.config:
            self.config["benchmark"]["post_model"]["name"] = post_model_path
            self.config["benchmark"]["post_model"]["type"] = model_type
            if model_type == "adapter":
                self.config["benchmark"]["post_model"]["base_model"] = ft_config.get("base_model", "unsloth/gemma-3n-E2B-it")
        
        # Save
        self.config["last_modified"] = datetime.now().isoformat()
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Post-model path updated: {post_model_path} ({model_type})")

    def load_or_create_config(self):
        """Loads existing config or creates new one"""
        if os.path.exists(self.config_file):
            print(f"📋 Loading existing configuration: {self.config_file}")
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # Update timestamp
            self.config["last_modified"] = datetime.now().isoformat()
            
            # In auto mode: No question, use existing config
            if self.auto_mode:
                print("🤖 Auto mode: Using existing configuration")
                # Check if post-model path should be updated
                if self.config.get("benchmark", {}).get("post_model", {}).get("name") in ["auto-detect", None, ""]:
                    print("\n🔍 Post-model path needs to be updated...")
                    self.update_post_model_path()
            else:
                # Interactive mode: Ask if reconfigure
                reconfigure = input("\n🔄 Create new configuration? (y/n) [n]: ").lower().strip() == 'y'
                if reconfigure:
                    return self.create_central_config()
                
                # Check if post-model path should be updated
                if self.config.get("benchmark", {}).get("post_model", {}).get("name") in ["auto-detect", None, ""]:
                    print("\n🔍 Post-model path needs to be updated...")
                    self.update_post_model_path()
            
            # Save model name in metrics
            self.metrics['model_name'] = self.config.get("finetuning", {}).get("model_name", "Unknown")
            
            return self.config
        else:
            if self.auto_mode:
                print("❌ No configuration found! Creating default configuration...")
                # Create default config without inputs
                return self.create_default_config()
            else:
                print("🆕 No configuration found - creating new one")
                return self.create_central_config()

    def setup_environment(self):
        """Sets environment variables from config"""
        if "tokens" in self.config:
            tokens = self.config["tokens"]
            if tokens.get("hf_token"):
                os.environ["HF_TOKEN"] = tokens["hf_token"]
                os.environ["HUGGINGFACE_TOKEN"] = tokens["hf_token"]
                print("🔑 HF_TOKEN set")
            if tokens.get("hf_write_token"):
                os.environ["HF_WRITE_TOKEN"] = tokens["hf_write_token"]
                print("🔑 HF_WRITE_TOKEN set")

    def copy_config_loader(self):
        """Copies config_loader.py to all modules"""
        # The config_loader code remains the same as before
        config_loader_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Central Config Loader for OpenTuneWeaver Pipeline
Loads the central pipeline_config.json for all modules
"""

import os
import json
import sys
from pathlib import Path

class PipelineConfigLoader:
    """Loads and manages the central pipeline configuration"""
    
    def __init__(self, module_id=None):
        """
        Initializes the config loader
        Args:
            module_id: Module ID (e.g. "01_convert", "02_genwiki", etc.)
        """
        self.module_id = module_id
        self.config = self._load_config()
        self.api_config = None
        
        if module_id and module_id in self.config.get("api_configs", {}):
            self.api_config = self.config["api_configs"][module_id]
    
    def _find_config_file(self):
        """Searches for the central config file"""
        # Priority 1: Environment variable
        if "PIPELINE_CONFIG_PATH" in os.environ:
            config_path = Path(os.environ["PIPELINE_CONFIG_PATH"])
            if config_path.exists():
                return config_path
        
        # Priority 2: Search relative to current directory
        search_paths = [
            Path.cwd() / "pipeline_config.json", # In current directory
            Path.cwd().parent / "pipeline_config.json", # One level up
            Path.cwd().parent.parent / "pipeline_config.json", # Two levels up
            Path.cwd().parent.parent.parent / "pipeline_config.json", # Three levels up (for modules)
            Path(__file__).parent / "pipeline_config.json", # Next to this script
            Path(__file__).parent.parent / "pipeline_config.json",
            Path(__file__).parent.parent.parent / "pipeline_config.json",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        # Not found
        raise FileNotFoundError(
            "pipeline_config.json not found!\\n"
            "Please run run_pipeline.py first or set PIPELINE_CONFIG_PATH"
        )
    
    def _load_config(self):
        """Loads the central configuration"""
        config_path = self._find_config_file()
        print(f"📋 Loading configuration from: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Set environment variables for tokens
            if "tokens" in config:
                if config["tokens"].get("hf_token"):
                    os.environ["HF_TOKEN"] = config["tokens"]["hf_token"]
                    os.environ["HUGGINGFACE_TOKEN"] = config["tokens"]["hf_token"]
                if config["tokens"].get("hf_write_token"):
                    os.environ["HF_WRITE_TOKEN"] = config["tokens"]["hf_write_token"]
            
            return config
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing config: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            sys.exit(1)
    
    def get_api_config(self, module_id=None):
        """
        Returns the API configuration for a module
        Args:
            module_id: Optional module ID, if not set in constructor
        Returns:
            Dict with API configuration
        """
        if module_id:
            return self.config.get("api_configs", {}).get(module_id, {})
        return self.api_config or {}
    
    def get_finetuning_config(self):
        """Returns the finetuning configuration"""
        return self.config.get("finetuning", {})
    
    def get_benchmark_config(self):
        """Returns the benchmark configuration"""
        return self.config.get("benchmark", {})
    
    def get_pipeline_config(self):
        """Returns the pipeline configuration"""
        return self.config.get("pipeline", {})
    
    def get_tokens(self):
        """Returns the token configuration"""
        return self.config.get("tokens", {})
    
    def get_full_config(self):
        """Returns the complete configuration"""
        return self.config
    
    def print_config_summary(self):
        """Prints a configuration summary"""
        print("\\n📊 Configuration Overview:")
        print(f" 📋 Version: {self.config.get('version', 'unknown')}")
        print(f" 📅 Created: {self.config.get('created', 'unknown')}")
        print(f" 🔄 Modified: {self.config.get('last_modified', 'unknown')}")
        
        if self.api_config:
            print(f"\\n 📡 API Config for {self.module_id}:")
            api_type = "OpenAI" if self.api_config.get("use_openai_api") else "Ollama"
            if api_type == "OpenAI":
                print(f" - Type: {api_type}")
                print(f" - URL: {self.api_config.get('openai_base_url', 'not set')}")
                print(f" - Model: {self.api_config.get('openai_model_name', 'not set')}")
            else:
                print(f" - Type: {api_type}")
                print(f" - URL: {self.api_config.get('ollama_server_url', 'not set')}")
                print(f" - Model: {self.api_config.get('ollama_model_name', 'not set')}")

# Convenience functions for direct import
def load_config(module_id=None):
    """
    Loads the central pipeline configuration
    Args:
        module_id: Optional module ID (e.g. "01_convert")
    Returns:
        PipelineConfigLoader instance
    """
    return PipelineConfigLoader(module_id)

def get_api_config(module_id):
    """
    Gets the API configuration directly for a module
    Args:
        module_id: Module ID (e.g. "01_convert")
    Returns:
        Dict with API configuration
    """
    loader = PipelineConfigLoader(module_id)
    return loader.get_api_config()
'''
        
        # Save config_loader.py in main directory
        loader_path = Path("config_loader.py")
        with open(loader_path, 'w', encoding='utf-8') as f:
            f.write(config_loader_content)
        
        print(f"✅ config_loader.py created: {loader_path}")
        
        # Optional: Copy to all modules
        for step in STEPS:
            if step['script']:  # Only for real modules (not step 8)
                module_dir = Path(step['working_dir'])
                if module_dir.exists():
                    target_path = module_dir / "config_loader.py"
                    shutil.copy2(loader_path, target_path)
                    print(f" 📋 Copied to: {target_path}")

    def copy_files_between_steps(self, current_step_id):
        """Copies output files between pipeline steps"""
        transfers = {
            2: {
                'source': 'modules/01_convert/OUTPUT',
                'target': 'modules/02_wiki/INPUT',
                'files': '*.md'
            },
            3: {
                'source': 'modules/02_wiki/OUTPUT',
                'target': 'modules/03_instructQA/INPUT',
                'files': 'lexikon_*.json'
            },
            4: {
                'source': 'modules/03_instructQA/OUTPUT',
                'target': 'modules/04_format/INPUT',
                'files': 'qa_instruct_dataset.json'
            },
            5: {
                'source': 'modules/02_wiki/OUTPUT',
                'target': 'modules/05_bmcreator/INPUT',
                'files': 'lexikon_*.json'
            },
            6: {
                'source': 'modules/04_format/OUTPUT',
                'target': 'modules/06_finetuning/INPUT',
                'files': 'dataset.json'
            },
            7: {
                'source': 'modules/05_bmcreator/BENCHMARKFRAGEN',
                'target': 'modules/07_benchmark/BENCHMARKFRAGEN',
                'files': 'benchmark_fragen_complete.json'
            }
        }
        
        if current_step_id not in transfers:
            return True
        
        transfer = transfers[current_step_id]
        source_dir = transfer['source']
        target_dir = transfer['target']
        file_pattern = transfer['files']
        
        print(f"🔄 Copying files: {source_dir} → {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        
        source_files = glob(f"{source_dir}/{file_pattern}")
        copied_count = 0
        
        for source_file in source_files:
            if os.path.isfile(source_file):
                filename = os.path.basename(source_file)
                target_file = os.path.join(target_dir, filename)
                try:
                    shutil.copy2(source_file, target_file)
                    copied_count += 1
                    print(f" ✅ {filename}")
                except Exception as e:
                    print(f" ❌ Error with {filename}: {e}")
        
        print(f" 📊 {copied_count} files copied")
        return copied_count > 0 or current_step_id == 1

    def save_metrics(self, output_dir=None):
        """Saves the collected metrics"""
        if output_dir is None:
            output_dir = '.'
        
        metrics_file = os.path.join(output_dir, 'pipeline_metrics.json')
        
        # Calculate final metrics
        self.metrics['end_time'] = datetime.now().isoformat()
        duration = datetime.now() - self.start_time
        self.metrics['total_duration'] = str(duration)
        self.metrics['total_duration_seconds'] = duration.total_seconds()
        
        # Save
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 Metrics saved: {metrics_file}")
        return metrics_file

    def transfer_and_archive_results(self):
        """
        Transfers all pipeline results to data/OUTPUT with timestamp-based archiving
        Creates a common ZIP with all documents and collects metrics
        """
        print(f"\n{'='*60}")
        print("📦 STEP 8: ARCHIVE AND TRANSFER RESULTS")
        print(f"{'='*60}")
        
        base_dir = Path('.').resolve()
        output_base = base_dir / 'data' / 'OUTPUT'
        
        # Ensure data/OUTPUT exists
        output_base.mkdir(parents=True, exist_ok=True)
        print(f"📁 Target directory: {output_base}")
        
        # Generate unique timestamp and get model name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = self.config.get("finetuning", {}).get("model_name", "model")
        archive_name = f"{model_name}_{timestamp}"
        print(f"🕒 Timestamp: {timestamp}")
        print(f"🤖 Model name: {model_name}")
        
        # Collect final data statistics before archiving
        print("\n📊 Collecting final statistics...")
        
        # Lexicon statistics
        self.metrics['data_statistics']['lexikon'] = self.analyze_lexikon_files('modules/02_wiki/OUTPUT')
        
        # QA dataset statistics
        self.metrics['data_statistics']['qa_dataset'] = self.analyze_qa_dataset('modules/03_instructQA/OUTPUT')
        
        # Formatted dataset statistics
        self.metrics['data_statistics']['formatted_dataset'] = self.analyze_qa_dataset('modules/04_format/OUTPUT')
        
        # Benchmark questions statistics
        self.metrics['data_statistics']['benchmark_questions'] = self.analyze_benchmark_questions('modules/05_bmcreator/BENCHMARKFRAGEN')
        
        # Training results
        self.metrics['data_statistics']['training'] = self.analyze_training_results('modules/06_finetuning')
        
        # Benchmark results
        self.metrics['benchmark_results'] = self.analyze_benchmark_results('modules/07_benchmark/OUTPUT')
        
        # Save metrics
        metrics_file = self.save_metrics()
        
        # Temporary directory for collecting all files
        temp_collection_dir = output_base / f"temp_{archive_name}"
        temp_collection_dir.mkdir(exist_ok=True)
        
        # Mapping of pipeline steps to source and target directories
        transfer_config = {
            'Step1_Convert': {
                'source': base_dir / 'modules' / '01_convert' / 'OUTPUT',
                'patterns': ['*.md'],
                'description': 'Converted Markdown files'
            },
            'Step2_Wiki': {
                'source': base_dir / 'modules' / '02_wiki' / 'OUTPUT', 
                'patterns': ['lexikon_*.json'],
                'description': 'Wiki lexicon JSON files'
            },
            'Step3_InstructQA': {
                'source': base_dir / 'modules' / '03_instructQA' / 'OUTPUT',
                'patterns': ['*.json'],
                'description': 'InstructQA datasets'
            },
            'Step4_Conversation': {
                'source': base_dir / 'modules' / '04_format' / 'OUTPUT',
                'patterns': ['*.json'],
                'description': 'Conversation-formatted datasets'
            },
            'Step5_Benchmark': {
                'sources': [
                    base_dir / 'modules' / '05_bmcreator' / 'BENCHMARKFRAGEN',
                    base_dir / 'modules' / '07_benchmark' / 'OUTPUT'
                ],
                'patterns': ['*.json', '*.txt', '*.csv', '*.md'],
                'description': 'Benchmark questions and results'
            },
            'Step6_PipelineMeta': {
                'sources': [base_dir],  # Main directory for pipeline_config.json
                'patterns': ['pipeline_config.json', 'pipeline_metrics.json', '*.log'],
                'description': 'Pipeline metadata incl. configuration and metrics'
            }
        }
        
        # Handle viewer separately (corrected path)
        viewer_config = {
            'Viewer': {
                'source': base_dir / 'viewer',  # Corrected path: directly in main directory
                'patterns': ['*.html', '*.js', '*.css'],
                'copy_directories': [
                    {'source': base_dir / 'viewer' / 'images', 'target_name': 'images'}
                ],
                'description': 'Viewer HTML and associated files'
            }
        }
        
        transferred_items = []
        total_file_count = 0
        
        # Step 1: Collect all documents in subdirectories
        print("\n📁 Collecting all documents...")
        
        for step_name, config in transfer_config.items():
            step_subdir = temp_collection_dir / step_name
            step_subdir.mkdir(exist_ok=True)
            
            file_count = 0
            sources = config.get('sources', [config.get('source')])
            
            for source_dir in sources:
                if source_dir and source_dir.exists():
                    for pattern in config['patterns']:
                        for file_path in source_dir.glob(pattern):
                            if file_path.is_file():
                                dest_path = step_subdir / file_path.name
                                shutil.copy2(file_path, dest_path)
                                file_count += 1
                                print(f"  ✅ {step_name}/{file_path.name}")
            
            if file_count > 0:
                transferred_items.append(f"{step_name}: {file_count} files")
                total_file_count += file_count
                print(f"  📊 {step_name}: {file_count} files collected")
        
        # Add viewer
        print("\n📁 Processing viewer...")
        viewer_source = viewer_config['Viewer']['source']
        
        if viewer_source.exists():
            viewer_subdir = temp_collection_dir / 'Viewer'
            viewer_subdir.mkdir(exist_ok=True)
            
            viewer_file_count = 0
            
            # HTML and other files
            for pattern in viewer_config['Viewer']['patterns']:
                for file_path in viewer_source.glob(pattern):
                    if file_path.is_file():
                        dest_path = viewer_subdir / file_path.name
                        shutil.copy2(file_path, dest_path)
                        viewer_file_count += 1
                        print(f"  ✅ Viewer/{file_path.name}")
            
            # Images directory
            for dir_config in viewer_config['Viewer']['copy_directories']:
                source_dir = dir_config['source']
                target_name = dir_config['target_name']
                
                if source_dir.exists() and source_dir.is_dir():
                    target_dir = viewer_subdir / target_name
                    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
                    
                    # Count files in copied directory
                    copied_files = list(target_dir.rglob('*'))
                    dir_file_count = len([f for f in copied_files if f.is_file()])
                    viewer_file_count += dir_file_count
                    print(f"  📂 Viewer/{target_name}: {dir_file_count} files")
            
            if viewer_file_count > 0:
                transferred_items.append(f"Viewer: {viewer_file_count} files")
                total_file_count += viewer_file_count
                print(f"  📊 Viewer: {viewer_file_count} files collected")
        else:
            print(f"  ⚠️ Viewer directory not found: {viewer_source}")
            print(f"     Expected path: viewer/ (relative to run_pipeline.py)")
        
        # Step 2: Create common ZIP archive
        if total_file_count > 0:
            documents_zip_path = output_base / f"{archive_name}_documents.zip"
            print(f"\n📦 Creating document archive: {documents_zip_path.name}")
            
            with zipfile.ZipFile(documents_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_collection_dir.rglob('*'):
                    if file_path.is_file():
                        archive_name_in_zip = file_path.relative_to(temp_collection_dir)
                        zipf.write(file_path, arcname=archive_name_in_zip)
            
            # Calculate size
            zip_size = documents_zip_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✅ Document archive created: {zip_size:.2f} MB")
            
            # Clean up temporary directory
            shutil.rmtree(temp_collection_dir)
        
        # Step 3: Archive CustomModel separately (corrected path)
        print(f"\n🤖 Processing CustomModel...")
        custom_model_dir = base_dir / 'modules' / '06_finetuning' / 'CustomModel'
        
        if custom_model_dir.exists() and any(custom_model_dir.iterdir()):
            model_zip_path = output_base / f"{archive_name}_models.zip"
            
            print(f"  📦 Creating model archive (may take several minutes)...")
            print(f"     Source: {custom_model_dir}")
            
            # Use ZIP_STORED for large model files (no compression)
            # as these are already compressed
            with zipfile.ZipFile(model_zip_path, 'w', zipfile.ZIP_STORED) as zipf:
                for file_path in custom_model_dir.rglob('*'):
                    if file_path.is_file():
                        archive_name_in_zip = file_path.relative_to(custom_model_dir.parent)
                        zipf.write(file_path, arcname=archive_name_in_zip)
                        # Show progress for large files
                        if file_path.stat().st_size > 100 * 1024 * 1024:  # > 100 MB
                            file_size_mb = file_path.stat().st_size / (1024 * 1024)
                            print(f"     📄 {file_path.name} ({file_size_mb:.1f} MB)")
            
            # Calculate size
            model_size = model_zip_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✅ Model archive created: {model_size:.1f} MB")
        else:
            print(f"  ⚠️ CustomModel directory not found or empty")
            print(f"     Expected path: {custom_model_dir}")
        
        # Update metrics with archive information
        self.metrics['archive_info'] = {
            'timestamp': timestamp,
            'model_name': model_name,
            'documents_archive': f"{archive_name}_documents.zip" if total_file_count > 0 else None,
            'models_archive': f"{archive_name}_models.zip" if custom_model_dir.exists() else None,
            'total_files': total_file_count
        }
        
        # Save final metrics
        final_metrics_file = output_base / f"{archive_name}_pipeline_metrics.json"
        with open(final_metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n📊 Final metrics saved: {final_metrics_file.name}")
        
        # Step 4: Clean up source directories
        print(f"\n🧹 CLEANING UP SOURCE DIRECTORIES")
        
        # In auto mode: Automatic cleanup if cleanup_after is set
        if self.auto_mode and self.cleanup_after:
            print("🤖 Auto mode: Performing automatic cleanup...")
            cleanup_confirm = 'y'
        else:
            cleanup_confirm = input("Clean up source directories now? (y/n) [y]: ").lower().strip()
        
        if cleanup_confirm != 'n':
            cleanup_dirs = [
                'modules/01_convert/OUTPUT',
                'modules/02_wiki/OUTPUT', 
                'modules/03_instructQA/OUTPUT',
                'modules/04_format/OUTPUT',
                'modules/05_bmcreator/BENCHMARKFRAGEN',
                'modules/07_benchmark/OUTPUT',
                'modules/06_finetuning/CustomModel'  # Corrected path
            ]
            
            cleaned_count = 0
            for cleanup_dir in cleanup_dirs:
                cleanup_path = base_dir / cleanup_dir
                if cleanup_path.exists():
                    # Special handling for CustomModel (only delete content, not the directory)
                    if 'CustomModel' in str(cleanup_path):
                        for item in cleanup_path.iterdir():
                            if item.is_file():
                                item.unlink()
                                cleaned_count += 1
                            elif item.is_dir():
                                shutil.rmtree(item)
                                cleaned_count += 1
                        print(f"  🗑️ CustomModel content cleaned")
                    else:
                        # For other directories: only delete files
                        for item in cleanup_path.iterdir():
                            if item.is_file():
                                item.unlink()
                                cleaned_count += 1
                                print(f"  🗑️ Deleted: {cleanup_dir}/{item.name}")
            
            print(f"  📊 Total {cleaned_count} items cleaned")
            print(f"  🔒 Viewer files preserved for further use")
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 ARCHIVING COMPLETED")
        print(f"{'='*60}")
        print(f"📁 Target directory: {output_base}")
        print(f"🤖 Model: {model_name}")
        print(f"🕒 Timestamp: {timestamp}")
        print(f"\n📦 Created archives:")
        
        # List all created archives
        for archive_file in output_base.glob(f"{model_name}_{timestamp}_*.zip"):
            archive_size = archive_file.stat().st_size / (1024 * 1024)
            print(f"  • {archive_file.name} ({archive_size:.1f} MB)")
        
        # List metrics files
        for metrics_file in output_base.glob(f"{model_name}_{timestamp}_*.json"):
            file_size = metrics_file.stat().st_size / 1024
            print(f"  • {metrics_file.name} ({file_size:.1f} KB)")
        
        # Calculate total size
        total_size = sum(
            f.stat().st_size for f in output_base.glob(f"{model_name}_{timestamp}_*")
        ) / (1024 * 1024)
        
        print(f"\n💾 Total size of all files: {total_size:.1f} MB")
        print(f"📊 Total {total_file_count} files archived")
        
        # Show important metrics
        if self.metrics.get('benchmark_results', {}).get('comparison_available'):
            print(f"\n🏆 Benchmark Results:")
            print(f"  Pre-Score: {self.metrics['benchmark_results']['pre_score']:.1f}%")
            print(f"  Post-Score: {self.metrics['benchmark_results']['post_score']:.1f}%")
            print(f"  Improvement: {self.metrics['benchmark_results']['improvement']:+.1f}%")
        
        print(f"{'='*60}")
        
        return True

    def run_step(self, step):
        """Executes a single pipeline step and collects metrics"""
        
        # Special handling for step 8 (archiving)
        if step['id'] == 8:
            step_start = datetime.now()
            success = self.transfer_and_archive_results()
            step_duration = (datetime.now() - step_start).total_seconds()
            
            self.metrics['steps'][f"step_{step['id']}"] = {
                'name': step['name'],
                'success': success,
                'duration_seconds': step_duration,
                'timestamp': datetime.now().isoformat()
            }
            
            return success
        
        print(f"\n{'='*60}")
        print(f"📋 STEP {step['id']}/8: {step['name'].upper()}")
        print(f"{'='*60}")

        self.write_status_update(step['id'], 'running')
        
        # Initialize step metrics
        step_metrics = {
            'name': step['name'],
            'start_time': datetime.now().isoformat(),
            'success': False,
            'duration_seconds': 0,
            'error': None,
            'files_processed': 0,
            'files_created': 0
        }
        
        step_start = datetime.now()
        
        # After fine-tuning: Update post-model path
        if step['id'] == 6: # After fine-tuning
            print("\n🔄 Updating post-model path after fine-tuning...")
            self.update_post_model_path()
        
        # Prepare working directory
        working_dir = step['working_dir']
        print(f"📁 Working directory: {working_dir}")
        
        # Create directories
        for subdir in step['directories']:
            dir_path = os.path.join(working_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            print(f" ✅ {dir_path}")
        
        # Count input files before step
        if 'INPUT' in step['directories']:
            input_dir = os.path.join(working_dir, 'INPUT')
            if os.path.exists(input_dir):
                step_metrics['files_processed'] = len(os.listdir(input_dir))
        
        # Copy files from previous step
        if step['id'] > 1:
            self.copy_files_between_steps(step['id'])
        
        # Set config path as environment variable
        env = os.environ.copy()
        env["PIPELINE_CONFIG_PATH"] = os.path.abspath(self.config_file)
        
        print(f"\n🚀 Starting {step['name']}...")
        print("=" * 60)
        
        try:
            result = subprocess.run(
                [sys.executable, os.path.basename(step['script'])],
                cwd=working_dir,
                env=env
            )
            
            print("=" * 60)
            
            step_duration = (datetime.now() - step_start).total_seconds()
            step_metrics['duration_seconds'] = step_duration
            
            if result.returncode == 0:
                print(f"✅ {step['name']} successful!")
                step_metrics['success'] = True
                
                # After fine-tuning: Update post-model path
                if step['id'] == 6:
                    self.update_post_model_path()
                
                # Count output files after step
                if 'OUTPUT' in step['directories']:
                    output_dir = os.path.join(working_dir, 'OUTPUT')
                    if os.path.exists(output_dir):
                        step_metrics['files_created'] = len(os.listdir(output_dir))
                elif 'BENCHMARKFRAGEN' in step['directories']:
                    output_dir = os.path.join(working_dir, 'BENCHMARKFRAGEN')
                    if os.path.exists(output_dir):
                        step_metrics['files_created'] = len(os.listdir(output_dir))
                
                # Capture specific metrics per step
                if step['id'] == 2:  # Wiki Generation
                    output_dir = os.path.join(working_dir, 'OUTPUT')
                    lexikon_stats = self.analyze_lexikon_files(output_dir)
                    step_metrics['lexikon_entries'] = lexikon_stats['total_entries']
                    step_metrics['lexikon_categories'] = len(lexikon_stats['categories'])
                
                elif step['id'] == 3:  # InstructQA
                    output_dir = os.path.join(working_dir, 'OUTPUT')
                    qa_stats = self.analyze_qa_dataset(output_dir)
                    step_metrics['qa_pairs'] = qa_stats['total_qa_pairs']
                
                elif step['id'] == 5:  # Benchmark Creation
                    benchmark_dir = os.path.join(working_dir, 'BENCHMARKFRAGEN')
                    bm_stats = self.analyze_benchmark_questions(benchmark_dir)
                    step_metrics['benchmark_questions'] = bm_stats['total_questions']
                    step_metrics['benchmark_categories'] = len(bm_stats['categories'])
                
                self.metrics['steps'][f"step_{step['id']}"] = step_metrics
                self.write_status_update(step['id'], 'completed', step_metrics, step_duration)
                return True
            else:
                print(f"❌ {step['name']} failed (Code: {result.returncode})")
                step_metrics['error'] = f"Return code: {result.returncode}"
                self.metrics['steps'][f"step_{step['id']}"] = step_metrics
                self.metrics['errors'].append(f"Step {step['id']} failed: {step['name']}")
                self.write_status_update(step['id'], 'failed')
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️ Cancelled")
            step_metrics['error'] = "User interrupted"
            step_metrics['duration_seconds'] = (datetime.now() - step_start).total_seconds()
            self.metrics['steps'][f"step_{step['id']}"] = step_metrics
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            step_metrics['error'] = str(e)
            step_metrics['duration_seconds'] = (datetime.now() - step_start).total_seconds()
            self.metrics['steps'][f"step_{step['id']}"] = step_metrics
            self.metrics['errors'].append(f"Step {step['id']} error: {str(e)}")
            return False

    def cleanup(self):
        """Offers cleanup of previous runs"""
        print("\n🧹 CLEANUP")
        print("="*60)
        
        dirs_to_clean = []
        for step in STEPS:
            if step['script']:  # Only for real modules (not step 8)
                for subdir in step['directories']:
                    dir_path = os.path.join(step['working_dir'], subdir)
                    if os.path.exists(dir_path) and os.listdir(dir_path):
                        dirs_to_clean.append(dir_path)
        
        if not dirs_to_clean:
            print("✅ No temporary files found")
            return
        
        print("Found temporary directories:")
        for dir_path in dirs_to_clean:
            file_count = len(os.listdir(dir_path))
            print(f" • {dir_path} ({file_count} files)")
        
        if input("\n🧹 Delete everything? (y/n) [n]: ").lower().strip() == 'y':
            for dir_path in dirs_to_clean:
                try:
                    shutil.rmtree(dir_path)
                    os.makedirs(dir_path, exist_ok=True)
                    print(f" ✅ Cleaned: {dir_path}")
                except Exception as e:
                    print(f" ❌ Error with {dir_path}: {e}")

    def run(self, start_step=1, end_step=8):
        """Runs the pipeline and collects metrics"""
        print("\n" + "="*60)
        print("🚀 OpenTuneWeaver Pipeline - SIMPLIFIED")
        print("="*60)
        print(f"📋 Version: 5.3 (with metrics collection)")
        print(f"⏰ Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📍 Steps: {start_step} to {end_step}")
        print("="*60)
        
        # Load/create config
        self.load_or_create_config()
        
        # Copy config loader
        if not os.path.exists("config_loader.py"):
            self.copy_config_loader()
        
        # Set up environment
        self.setup_environment()
        
        # Offer cleanup
        if self.config.get("pipeline", {}).get("auto_cleanup", False):
            self.cleanup()
        
        # Execute steps
        completed = []
        failed = []
        
        for step in STEPS:
            if step['id'] < start_step or step['id'] > end_step:
                continue
                
            success = self.run_step(step)
            
            if success:
                completed.append(step['name'])
            else:
                failed.append(step['name'])
                if not self.config.get("pipeline", {}).get("continue_on_error", True):
                    print("❌ Pipeline stopped due to error")
                    break
                
                if input("Continue anyway? (y/n) [y]: ").lower().strip() == 'n':
                    break
            
            if step['id'] < end_step:
                time.sleep(2)
        
        # Final metrics
        self.metrics['completed_steps'] = completed
        self.metrics['failed_steps'] = failed
        
        # Summary
        duration = datetime.now() - self.start_time
        print(f"\n{'='*60}")
        print("📊 PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"⏱️ Duration: {duration}")
        print(f"✅ Successful: {len(completed)} steps")
        
        if completed:
            for name in completed:
                print(f" • {name}")
        
        if failed:
            print(f"❌ Failed: {len(failed)} steps")
            for name in failed:
                print(f" • {name}")
        
        # Show important metrics
        if self.metrics.get('data_statistics'):
            print(f"\n📊 DATA STATISTICS:")
            stats = self.metrics['data_statistics']
            
            if 'lexikon' in stats and stats['lexikon']:
                print(f" • Lexicon entries: {stats['lexikon'].get('total_entries', 0)}")
            if 'qa_dataset' in stats and stats['qa_dataset']:
                print(f" • QA pairs created: {stats['qa_dataset'].get('total_qa_pairs', 0)}")
            if 'benchmark_questions' in stats and stats['benchmark_questions']:
                print(f" • Benchmark questions: {stats['benchmark_questions'].get('total_questions', 0)}")
            if 'training' in stats and stats['training'].get('model_trained'):
                print(f" • Model trained: ✅")
                if stats['training'].get('merged_model_available'):
                    print(f"   - Merged Model: ✅")
                if stats['training'].get('gguf_available'):
                    print(f"   - GGUF Export: ✅")
        
        if self.metrics.get('benchmark_results', {}).get('comparison_available'):
            print(f"\n🏆 BENCHMARK RESULTS:")
            br = self.metrics['benchmark_results']
            print(f" • Pre-Finetuning: {br['pre_score']:.1f}%")
            print(f" • Post-Finetuning: {br['post_score']:.1f}%")
            print(f" • Improvement: {br['improvement']:+.1f}%")
        
        print(f"{'='*60}")
        print("✨ Pipeline completed!")
        
        # Save final metrics
        if self.config.get("pipeline", {}).get("save_metrics", True):
            self.save_metrics()

def main():
    """Main function with support for automated mode"""
    import argparse
    
    # Command line arguments for automation
    parser = argparse.ArgumentParser(description='OpenTuneWeaver Pipeline Runner')
    parser.add_argument('--auto', action='store_true', 
                       help='Automated mode (no input required)')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'data', 'training', 'single', 'custom', 'archive', 'cleanup'],
                       help='Pipeline mode: full(1-8), data(1-5), training(6-7), single, custom, archive(8), cleanup')
    parser.add_argument('--step', type=int, default=1,
                       help='Single step (for mode=single)')
    parser.add_argument('--start', type=int, default=1,
                       help='Start step (for mode=custom)')
    parser.add_argument('--end', type=int, default=8,
                       help='End step (for mode=custom)')
    parser.add_argument('--use-existing-config', action='store_true', default=True,
                       help='Use existing configuration without asking')
    parser.add_argument('--cleanup-after', action='store_true', default=True,
                       help='Automatic cleanup after pipeline')
    
    args = parser.parse_args()
    
    runner = SimplifiedPipelineRunner()
    
    # Set flags for automated mode
    if args.auto:
        runner.auto_mode = True
        runner.use_existing_config = args.use_existing_config
        runner.cleanup_after = args.cleanup_after
        
        print("\n" + "="*60)
        print("🤖 OpenTuneWeaver Pipeline Runner - AUTOMATED MODE")
        print("="*60)
        print(f"📋 Mode: {args.mode}")
        print(f"✅ Use existing config: {args.use_existing_config}")
        print(f"🧹 Cleanup after pipeline: {args.cleanup_after}")
        print("="*60)
        
        # Run pipeline based on mode
        if args.mode == 'full':
            runner.run(1, 8)
        elif args.mode == 'data':
            runner.run(1, 5)
        elif args.mode == 'training':
            runner.run(6, 7)
        elif args.mode == 'single':
            runner.run(args.step, args.step)
        elif args.mode == 'custom':
            runner.run(args.start, args.end)
        elif args.mode == 'archive':
            runner.run(8, 8)
        elif args.mode == 'cleanup':
            runner.cleanup()
    else:
        # Interactive mode (original)
        print("\n" + "="*60)
        print("🎯 OpenTuneWeaver Pipeline Runner")
        print("="*60)
        
        print("\n📋 Pipeline Options:")
        print("1. 🔄 Complete pipeline with archiving (1-8)")
        print("2. 📝 Data processing only (1-5)")
        print("3. 🤖 Training & benchmark only (6-7)")
        print("4. ⚙️ Single step")
        print("5. 🎯 Custom")
        print("6. 📦 Archiving only (8)")
        print("7. 🧹 Cleanup")
        
        choice = input("\nOption (1-7) [1]: ").strip() or "1"
        
        if choice == "1":
            runner.run(1, 8)
        elif choice == "2":
            runner.run(1, 5)
        elif choice == "3":
            runner.run(6, 7)
        elif choice == "4":
            step_num = int(input("Which step (1-8)? "))
            runner.run(step_num, step_num)
        elif choice == "5":
            start = int(input("Start step [1]: ").strip() or "1")
            end = int(input("End step [8]: ").strip() or "8")
            runner.run(start, end)
        elif choice == "6":
            runner.run(8, 8)
        elif choice == "7":
            runner.cleanup()
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)