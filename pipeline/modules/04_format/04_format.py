#!/usr/bin/env python3
"""
04_format.py - Konvertiert qa_instruct_dataset.json zu dataset.json im korrekten Format für Fine-tuning
"""

import json
import os
from pathlib import Path

def load_qa_dataset(file_path):
    """Loads the QA dataset file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found!")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON file: {e}")
        return None

def format_conversation_entry(qa_entry):
    """Formats a single QA entry for fine-tuning"""
    instruction = qa_entry.get("instruction", "").strip()
    output = qa_entry.get("output", "").strip()
    
    if not instruction or not output:
        return None
    
    # Generate Gemini format for text field
    text_content = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>\n"
    
    # Generate structured conversation format
    conversation = {
        "conversations": [
            {
                "content": [{"text": instruction, "type": "text"}],
                "role": "user"
            },
            {
                "content": [{"text": output, "type": "text"}],
                "role": "assistant"  # IMPORTANT: "assistant" instead of "model"!
            }
        ],
        "text": text_content
    }
    
    return conversation

def convert_dataset(input_file, output_file):
    """Konvertiert den gesamten Datensatz"""
    print(f"📂 Lade QA-Datensatz von: {input_file}")
    
    # Lade die Eingabedatei
    qa_data = load_qa_dataset(input_file)
    if qa_data is None:
        return False
    
    # Extrahiere die Dateneinträge
    qa_entries = qa_data.get("data", [])
    if not qa_entries:
        print("❌ Keine Dateneinträge im 'data'-Feld gefunden!")
        return False
    
    print(f"📊 Gefunden: {len(qa_entries)} QA-Paare")
    
    # Konvertiere jeden Eintrag
    formatted_conversations = []
    skipped_count = 0
    
    for i, qa_entry in enumerate(qa_entries):
        formatted_entry = format_conversation_entry(qa_entry)
        if formatted_entry:
            formatted_conversations.append(formatted_entry)
        else:
            skipped_count += 1
            print(f"⚠️  Überspringe Eintrag {i+1}: Fehlende instruction oder output")
    
    print(f"✅ Erfolgreich formatiert: {len(formatted_conversations)} Einträge")
    if skipped_count > 0:
        print(f"⚠️  Übersprungen: {skipped_count} Einträge")
    
    # Speichere die formatierten Daten
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in formatted_conversations:
                json.dump(entry, f, ensure_ascii=False, separators=(',', ':'))
                f.write('\n')
        
        print(f"💾 Datensatz gespeichert: {output_file}")
        print(f"📊 Anzahl Einträge: {len(formatted_conversations)}")
        return True
        
    except Exception as e:
        print(f"❌ Fehler beim Speichern: {e}")
        return False

def main():
    """Hauptfunktion"""
    print("🔄 Starte Datensatz-Konvertierung...")
    
    # Dateipfade
    input_file = "INPUT/qa_instruct_dataset.json"
    output_file = "OUTPUT/dataset.json"
    
    # Erstelle OUTPUT-Ordner falls er nicht existiert
    os.makedirs("OUTPUT", exist_ok=True)
    
    # Prüfe ob Eingabedatei existiert
    if not os.path.exists(input_file):
        print(f"❌ Eingabedatei nicht gefunden: {input_file}")
        print("📁 Stelle sicher, dass sich die Datei im aktuellen Verzeichnis befindet.")
        return
    
    # Konvertiere den Datensatz
    success = convert_dataset(input_file, output_file)
    
    if success:
        print("\n🎉 Konvertierung erfolgreich abgeschlossen!")
        print(f"📁 Output-Datei: {output_file}")
        print("\n📋 Nächste Schritte:")
        print("   1. Überprüfe die dataset.json Datei")
        print("   2. Starte das Fine-tuning mit der neuen Datei")
    else:
        print("\n❌ Konvertierung fehlgeschlagen!")

if __name__ == "__main__":
    main()