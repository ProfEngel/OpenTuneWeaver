#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zentrale Config-Loader für OpenTuneWeaver Pipeline
Lädt die zentrale pipeline_config.json
"""

import os
import json
import sys
from pathlib import Path

class PipelineConfigLoader:
    """Lädt und verwaltet die zentrale Pipeline-Konfiguration"""
    
    @staticmethod
    def _normalize_api_url(url: str) -> str:
        """Ensures the API base URL ends with /v1 for OpenAI-compatible endpoints.
        
        Users often enter bare Ollama URLs like http://host:11434 without /v1,
        but all modules build endpoints as {API_BASE_URL}/chat/completions.
        This auto-appends /v1 if missing.
        """
        if not url:
            return url
        url = url.rstrip('/')
        if not url.endswith('/v1'):
            url = f"{url}/v1"
        return url
    
    def __init__(self):
        """Initialisiert den Config-Loader"""
        self.config = self._load_config()
        self.vision_config = self.config.get("vision", {})
        self.llm_config = self.config.get("llm", {})
        
        # Normalize API URLs
        if "api_base_url" in self.vision_config:
            self.vision_config["api_base_url"] = self._normalize_api_url(
                self.vision_config["api_base_url"]
            )
        if "api_base_url" in self.llm_config:
            self.llm_config["api_base_url"] = self._normalize_api_url(
                self.llm_config["api_base_url"]
            )
    
    def _find_config_file(self):
        """Sucht die zentrale Config-Datei"""
        # Priorität 1: Umgebungsvariable
        if "PIPELINE_CONFIG_PATH" in os.environ:
            config_path = Path(os.environ["PIPELINE_CONFIG_PATH"])
            if config_path.exists():
                return config_path
        
        # Priorität 2: Suche relativ zum aktuellen Verzeichnis
        search_paths = [
            Path.cwd() / "pipeline_config.json",
            Path.cwd().parent / "pipeline_config.json",
            Path.cwd().parent.parent / "pipeline_config.json",
            Path.cwd().parent.parent.parent / "pipeline_config.json",
            Path(__file__).parent / "pipeline_config.json",
            Path(__file__).parent.parent / "pipeline_config.json",
            Path(__file__).parent.parent.parent / "pipeline_config.json",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(
            "pipeline_config.json nicht gefunden!\n"
            "Bitte speichern Sie zuerst die Einstellungen im UI."
        )
    
    def _load_config(self):
        """Lädt die zentrale Konfiguration"""
        config_path = self._find_config_file()
        print(f"📋 Lade Konfiguration aus: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
            
        except json.JSONDecodeError as e:
            print(f"❌ Fehler beim Parsen der Config: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Fehler beim Laden der Config: {e}")
            sys.exit(1)
    
    def get_vision_config(self):
        """Gibt die Vision-Konfiguration zurück"""
        return self.vision_config

    def get_llm_config(self):
        """Gibt die LLM-Konfiguration zurück"""
        return self.llm_config
    
    def get_pipeline_config(self):
        """Gibt die Pipeline-Konfiguration zurück"""
        return self.config.get("pipeline", {})
    
    def get_module_config(self, module_name: str):
        """Returns a merged config for a specific module.
        
        Starts from the global vision/llm config and overlays any
        per-module overrides from api_configs.
        """
        # Pick base: 01_convert uses vision, everything else uses llm
        if module_name == "01_convert":
            base = dict(self.vision_config)
        else:
            base = dict(self.llm_config)
        
        # Check backward compatible names
        alt_names = [module_name]
        if module_name == "03_generate_qa":
            alt_names.append("03_instructQA")
        elif module_name == "03_instructQA":
            alt_names.append("03_generate_qa")
            
        module_overrides = {}
        for name in alt_names:
            if name in self.config.get("api_configs", {}):
                module_overrides = self.config.get("api_configs", {}).get(name, {})
                break
                
        if isinstance(module_overrides, dict):
            # Map the openai_* keys to the canonical names
            key_map = {
                "openai_base_url": "api_base_url",
                "openai_api_key": "api_key",
                "openai_model_name": "model_name",
            }
            for old_key, new_key in key_map.items():
                if old_key in module_overrides and module_overrides[old_key]:
                    value = module_overrides[old_key]
                    # Normalize API URLs
                    if new_key == "api_base_url":
                        value = self._normalize_api_url(value)
                    base[new_key] = value
            if "temperature" in module_overrides:
                base["temperature"] = module_overrides["temperature"]
        
        return base
    
    def get_full_config(self):
        """Gibt die komplette Konfiguration zurück"""
        return self.config
    
    def print_config_summary(self):
        """Gibt eine Zusammenfassung der Konfiguration aus"""
        print("\n📊 Konfigurations-Übersicht:")
        print(f"  📋 Version: {self.config.get('version', 'unbekannt')}")
        
        print(f"\n  👁️ Vision Config:")
        print(f"    - URL: {self.vision_config.get('api_base_url', 'nicht gesetzt')}")
        print(f"    - Model: {self.vision_config.get('model_name', 'nicht gesetzt')}")

        print(f"\n  🤖 LLM Config:")
        print(f"    - URL: {self.llm_config.get('api_base_url', 'nicht gesetzt')}")
        print(f"    - Model: {self.llm_config.get('model_name', 'nicht gesetzt')}")


# Convenience-Funktionen für direkten Import
def load_config():
    """
    Lädt die zentrale Pipeline-Konfiguration
    """
    return PipelineConfigLoader()

def get_vision_config():
    loader = PipelineConfigLoader()
    return loader.get_vision_config()

def get_llm_config():
    loader = PipelineConfigLoader()
    return loader.get_llm_config()

# Beispiel-Verwendung
if __name__ == "__main__":
    print("🧪 Teste Config-Loader...")
    try:
        loader = PipelineConfigLoader()
        loader.print_config_summary()
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Fehler: {e}")