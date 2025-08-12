#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zentrale Config-Loader für OpenTuneWeaver Pipeline
Lädt die zentrale pipeline_config.json für alle Module
"""

import os
import json
import sys
from pathlib import Path

class PipelineConfigLoader:
    """Lädt und verwaltet die zentrale Pipeline-Konfiguration"""
    
    def __init__(self, module_id=None):
        """
        Initialisiert den Config-Loader
        
        Args:
            module_id: ID des Moduls (z.B. "01_convert", "02_genwiki", etc.)
        """
        self.module_id = module_id
        self.config = self._load_config()
        self.api_config = None
        
        if module_id and module_id in self.config.get("api_configs", {}):
            self.api_config = self.config["api_configs"][module_id]
    
    def _find_config_file(self):
        """Sucht die zentrale Config-Datei"""
        # Priorität 1: Umgebungsvariable
        if "PIPELINE_CONFIG_PATH" in os.environ:
            config_path = Path(os.environ["PIPELINE_CONFIG_PATH"])
            if config_path.exists():
                return config_path
        
        # Priorität 2: Suche relativ zum aktuellen Verzeichnis
        search_paths = [
            Path.cwd() / "pipeline_config.json",  # Im aktuellen Verzeichnis
            Path.cwd().parent / "pipeline_config.json",  # Eine Ebene höher
            Path.cwd().parent.parent / "pipeline_config.json",  # Zwei Ebenen höher
            Path.cwd().parent.parent.parent / "pipeline_config.json",  # Drei Ebenen höher (für Module)
            Path(__file__).parent / "pipeline_config.json",  # Neben diesem Skript
            Path(__file__).parent.parent / "pipeline_config.json",
            Path(__file__).parent.parent.parent / "pipeline_config.json",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        # Nicht gefunden
        raise FileNotFoundError(
            "pipeline_config.json nicht gefunden!\n"
            "Bitte führen Sie zuerst run_pipeline.py aus oder setzen Sie PIPELINE_CONFIG_PATH"
        )
    
    def _load_config(self):
        """Lädt die zentrale Konfiguration"""
        config_path = self._find_config_file()
        print(f"📋 Lade Konfiguration aus: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Setze Umgebungsvariablen für Tokens
            if "tokens" in config:
                if config["tokens"].get("hf_token"):
                    os.environ["HF_TOKEN"] = config["tokens"]["hf_token"]
                    os.environ["HUGGINGFACE_TOKEN"] = config["tokens"]["hf_token"]
                if config["tokens"].get("hf_write_token"):
                    os.environ["HF_WRITE_TOKEN"] = config["tokens"]["hf_write_token"]
            
            return config
            
        except json.JSONDecodeError as e:
            print(f"❌ Fehler beim Parsen der Config: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Fehler beim Laden der Config: {e}")
            sys.exit(1)
    
    def get_api_config(self, module_id=None):
        """
        Gibt die API-Konfiguration für ein Modul zurück
        
        Args:
            module_id: Optionale Module-ID, falls nicht im Konstruktor gesetzt
        
        Returns:
            Dict mit API-Konfiguration
        """
        if module_id:
            return self.config.get("api_configs", {}).get(module_id, {})
        return self.api_config or {}
    
    def get_finetuning_config(self):
        """Gibt die Finetuning-Konfiguration zurück"""
        return self.config.get("finetuning", {})
    
    def get_benchmark_config(self):
        """Gibt die Benchmark-Konfiguration zurück"""
        return self.config.get("benchmark", {})
    
    def get_pipeline_config(self):
        """Gibt die Pipeline-Konfiguration zurück"""
        return self.config.get("pipeline", {})
    
    def get_tokens(self):
        """Gibt die Token-Konfiguration zurück"""
        return self.config.get("tokens", {})
    
    def get_full_config(self):
        """Gibt die komplette Konfiguration zurück"""
        return self.config
    
    def print_config_summary(self):
        """Gibt eine Zusammenfassung der Konfiguration aus"""
        print("\n📊 Konfigurations-Übersicht:")
        print(f"  📋 Version: {self.config.get('version', 'unbekannt')}")
        print(f"  📅 Erstellt: {self.config.get('created', 'unbekannt')}")
        print(f"  🔄 Geändert: {self.config.get('last_modified', 'unbekannt')}")
        
        if self.api_config:
            print(f"\n  📡 API-Config für {self.module_id}:")
            api_type = "OpenAI" if self.api_config.get("use_openai_api") else "Ollama"
            if api_type == "OpenAI":
                print(f"    - Typ: {api_type}")
                print(f"    - URL: {self.api_config.get('openai_base_url', 'nicht gesetzt')}")
                print(f"    - Model: {self.api_config.get('openai_model_name', 'nicht gesetzt')}")
            else:
                print(f"    - Typ: {api_type}")
                print(f"    - URL: {self.api_config.get('ollama_server_url', 'nicht gesetzt')}")
                print(f"    - Model: {self.api_config.get('ollama_model_name', 'nicht gesetzt')}")


# Convenience-Funktionen für direkten Import
def load_config(module_id=None):
    """
    Lädt die zentrale Pipeline-Konfiguration
    
    Args:
        module_id: Optionale Module-ID (z.B. "01_convert")
    
    Returns:
        PipelineConfigLoader Instanz
    """
    return PipelineConfigLoader(module_id)


def get_api_config(module_id):
    """
    Holt direkt die API-Konfiguration für ein Modul
    
    Args:
        module_id: Module-ID (z.B. "01_convert")
    
    Returns:
        Dict mit API-Konfiguration
    """
    loader = PipelineConfigLoader(module_id)
    return loader.get_api_config()


# Beispiel-Verwendung
if __name__ == "__main__":
    # Test: Lade Config
    print("🧪 Teste Config-Loader...")
    
    try:
        # Lade für ein spezifisches Modul
        loader = PipelineConfigLoader("02_genwiki")
        loader.print_config_summary()
        
        # Hole API-Config
        api_config = loader.get_api_config()
        print(f"\n📡 API-Konfiguration:")
        print(json.dumps(api_config, indent=2))
        
        # Hole Tokens
        tokens = loader.get_tokens()
        if tokens.get("hf_token"):
            print(f"\n🔑 HF-Token gefunden: {tokens['hf_token'][:8]}...")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Fehler: {e}")