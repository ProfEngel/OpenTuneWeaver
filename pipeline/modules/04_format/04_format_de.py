import json
import os
import sys
from pathlib import Path

# ========================================
# ZENTRALE KONFIGURATION LADEN
# ========================================
sys.path.append(str(Path(__file__).parent.parent.parent))  # Zum Hauptverzeichnis
from config_loader import PipelineConfigLoader

# Lade Konfiguration (04_format benötigt keine API-Config, aber wir laden trotzdem für Konsistenz)
config_loader = PipelineConfigLoader()
pipeline_config = config_loader.get_pipeline_config()

# Zeige geladene Konfiguration
print("=" * 60)
print("📋 KONFIGURATION GELADEN (04_format)")
print("=" * 60)
print("  📋 Format-Skript - Keine API benötigt")
print("  ✅ Pipeline-Config geladen")
print("=" * 60)

def create_gemma3_conversation(instruction, output):
    """Erstellt eine Conversation im Gemma-3 Format"""
    conversation = [
        {
            "content": [
                {
                    "text": instruction,
                    "type": "text"
                }
            ],
            "role": "user"
        },
        {
            "content": [
                {
                    "text": output,
                    "type": "text"
                }
            ],
            "role": "model"
        }
    ]
    return conversation

def create_gemma3_text(instruction, output):
    """Erstellt den formatierten Text für Gemma-3 Training"""
    text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>\n"
    return text

def process_qa_dataset():
    # Verzeichnispfade definieren
    input_dir = Path("INPUT")
    output_dir = Path("OUTPUT")
    
    # OUTPUT-Verzeichnis erstellen, falls es nicht existiert
    output_dir.mkdir(exist_ok=True)
    
    # Alle JSON-Dateien im INPUT-Ordner finden
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print("Keine JSON-Dateien im INPUT-Ordner gefunden.")
        return
    
    # Sammler für alle Datensätze
    all_entries = []
    
    # Jede JSON-Datei verarbeiten
    for json_file in json_files:
        print(f"Verarbeite {json_file.name}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Daten aus dem qa_instruct_dataset Format extrahieren
            if 'data' in data:
                qa_pairs = data['data']
                
                # Jedes Q&A-Paar konvertieren
                for qa in qa_pairs:
                    instruction = qa.get("instruction", "")
                    output = qa.get("output", "")
                    
                    # Leere Einträge überspringen
                    if not instruction.strip() or not output.strip():
                        continue
                    
                    # Gemma-3 Format erstellen
                    conversations = create_gemma3_conversation(instruction, output)
                    text = create_gemma3_text(instruction, output)
                    
                    # Eintrag im gewünschten Format
                    entry = {
                        "conversations": conversations,
                        "text": text
                    }
                    
                    all_entries.append(entry)
                
                print(f"  {len(qa_pairs)} Q&A-Paare aus {json_file.name} verarbeitet")
            
        except json.JSONDecodeError as e:
            print(f"Fehler beim Lesen von {json_file.name}: {e}")
        except Exception as e:
            print(f"Unerwarteter Fehler bei {json_file.name}: {e}")
    
    # Dataset als JSONL speichern (eine JSON-Zeile pro Eintrag)
    output_file = output_dir / "dataset.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in all_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"\nErfolgreich gespeichert: {output_file}")
        print(f"Insgesamt {len(all_entries)} Einträge im Gemma-3 Format erstellt")
        
        # Beispiel-Ausgabe zeigen
        if all_entries:
            print("\nBeispiel-Eintrag:")
            example = all_entries[0]
            print(f"Conversations: {json.dumps(example['conversations'], ensure_ascii=False, indent=2)}")
            print(f"Text: {repr(example['text'][:100])}...")
            
    except Exception as e:
        print(f"Fehler beim Speichern: {e}")

if __name__ == "__main__":
    print("🚀 Starte Dataset-Formatierung (04_format)")
    print("📝 Konvertiere QA-Paare zu Gemma-3 Format")
    print("📂 Input: INPUT/")
    print("📂 Output: OUTPUT/dataset.json")
    print("")
    process_qa_dataset()