import json
import requests
import random
import os
import sys
from pathlib import Path

# ========================================
# ZENTRALE KONFIGURATION LADEN
# ========================================
sys.path.append(str(Path(__file__).parent.parent.parent))  # Zum Hauptverzeichnis
from config_loader import PipelineConfigLoader

# Lade Konfiguration für dieses Modul
config_loader = PipelineConfigLoader("03_instructQA")
config = config_loader.get_api_config()

# Extrahiere Konfigurationswerte
USE_OPENAI_API = config.get("use_openai_api", True)
OPENAI_BASE_URL = config.get("openai_base_url", "http://localhost:11434/v1")
OPENAI_API_KEY = config.get("openai_api_key", "ollama")
OPENAI_MODEL_NAME = config.get("openai_model_name", "gemma3:12b-it-qat")
OLLAMA_SERVER_URL = config.get("ollama_server_url", "http://localhost:11434")
OLLAMA_API_KEY = config.get("ollama_api_key", "ollama")
OLLAMA_MODEL_NAME = config.get("ollama_model_name", "gemma3:12b-it-qat")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_SERVER_URL}/api/chat"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_SERVER_URL}/api/tags"

# Verzeichnisse
INPUT_DIR = "INPUT"
OUTPUT_DIR = "OUTPUT"
OUTPUT_FILENAME = "qa_instruct_dataset.json"

# Zeige geladene Konfiguration
print("=" * 60)
print("📋 KONFIGURATION GELADEN (03_instructQA)")
print("=" * 60)
config_loader.print_config_summary()
print("=" * 60)

# Verschiedene Fragetypen für abwechslungsreiche QA-Paare
QUESTION_TYPES = [
    {
        "type": "definition",
        "templates": [
            "Was ist {title}?",
            "Können Sie {title} erklären?",
            "Definieren Sie {title}.",
            "Was versteht man unter {title}?",
            "Erläutern Sie den Begriff {title}."
        ]
    },
    {
        "type": "detailed_explanation",
        "templates": [
            "Beschreiben Sie {title} ausführlich.",
            "Geben Sie eine detaillierte Erklärung von {title}.",
            "Was sollte man über {title} wissen?",
            "Erklären Sie mir {title} genauer.",
            "Welche wichtigen Aspekte hat {title}?"
        ]
    },
    {
        "type": "characteristics",
        "templates": [
            "Welche Merkmale hat {title}?",
            "Was sind die Haupteigenschaften von {title}?",
            "Welche Charakteristika zeichnen {title} aus?",
            "Was sind die wesentlichen Eigenschaften von {title}?",
            "Durch welche Besonderheiten ist {title} gekennzeichnet?"
        ]
    },
    {
        "type": "application",
        "templates": [
            "Wofür wird {title} verwendet?",
            "Welche Anwendungen hat {title}?",
            "In welchen Bereichen spielt {title} eine Rolle?",
            "Wo findet {title} Verwendung?",
            "Welche praktische Bedeutung hat {title}?"
        ]
    },
    {
        "type": "context",
        "templates": [
            "In welchem Kontext ist {title} relevant?",
            "Warum ist {title} wichtig?",
            "Welche Bedeutung hat {title}?",
            "In welchem Zusammenhang steht {title}?",
            "Welche Rolle spielt {title}?"
        ]
    }
]

def check_api_connection():
    """Überprüft API-Verbindung (OpenAI oder Ollama)."""
    if USE_OPENAI_API:
        return check_openai_connection()
    else:
        return check_ollama_connection()

def check_openai_connection():
    """Überprüft OpenAI-API-Verbindung."""
    try:
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": OPENAI_MODEL_NAME,
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 5
        }
        
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions", 
            json=payload, 
            headers=headers, 
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ OpenAI-API-Verbindung erfolgreich ({OPENAI_BASE_URL})")
            print(f"✅ Modell '{OPENAI_MODEL_NAME}' ist verfügbar")
            return True
        else:
            print(f"❌ OpenAI-API nicht erreichbar (Status: {response.status_code})")
            if response.status_code == 401:
                print("🔒 Authentifizierung fehlgeschlagen - prüfen Sie OPENAI_API_KEY")
            elif response.status_code == 404:
                print("❌ Modell nicht gefunden - prüfen Sie OPENAI_MODEL_NAME")
            return False
            
    except requests.RequestException as e:
        print(f"❌ OpenAI-API-Verbindung fehlgeschlagen: {e}")
        print(f"💡 Überprüfen Sie: Server läuft auf {OPENAI_BASE_URL}?")
        return False

def check_ollama_connection():
    """Überprüft Ollama-API-Verbindung."""
    try:
        headers = {
            'Authorization': f'Bearer {OLLAMA_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(f"{OLLAMA_SERVER_URL}/api/tags", headers=headers, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            print(f"✅ Ollama-Verbindung erfolgreich ({OLLAMA_SERVER_URL})")
            print(f"📋 Verfügbare Modelle: {', '.join(model_names[:3])}..." if len(model_names) > 3 else f"📋 Verfügbare Modelle: {', '.join(model_names)}")
            
            if OLLAMA_MODEL_NAME in model_names:
                print(f"✅ Modell '{OLLAMA_MODEL_NAME}' ist verfügbar")
                return True
            else:
                print(f"❌ Modell '{OLLAMA_MODEL_NAME}' nicht gefunden!")
                return False
        else:
            print(f"❌ Ollama nicht erreichbar (Status: {response.status_code})")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Ollama-Verbindung fehlgeschlagen: {e}")
        return False

def load_lexikon_files(input_dir):
    """Lädt alle Lexikon-JSON-Dateien aus dem Input-Verzeichnis."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input-Verzeichnis '{input_dir}' existiert nicht!")
        return []
    
    lexikon_files = list(input_path.glob("lexikon_*.json"))
    print(f"📁 Gefundene Lexikon-Dateien: {len(lexikon_files)}")
    
    all_entries = []
    
    for file in lexikon_files:
        print(f"📖 Lade: {file.name}")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entries = data.get('lexikon_entries', [])
                print(f"   - {len(entries)} Einträge gefunden")
                all_entries.extend(entries)
        except Exception as e:
            print(f"❌ Fehler beim Laden von {file.name}: {e}")
    
    print(f"📊 Gesamt: {len(all_entries)} Lexikon-Einträge")
    return all_entries

def generate_qa_prompt(title, lexikon_entry, question_type):
    """Generiert einen Prompt für die QA-Erstellung mit Markdown-Formatierung."""
    prompt = f"""
Erstelle ein Frage-Antwort-Paar für einen Instruct-Datensatz zum Thema "{title}".

Lexikon-Eintrag:
{lexikon_entry}

Aufgabe:
1. Erstelle eine {question_type["type"]}-Frage basierend auf dem Titel "{title}"
2. Beantworte die Frage ausführlich basierend auf dem Lexikon-Eintrag
3. Die Antwort soll 2-5 Sätze enthalten und informativ sein
4. Stil: Natürlich, lehrreich, wie ein Experte
5. Sprache: Deutsch

WICHTIG - MARKDOWN-FORMATIERUNG:
- Behalte ALLE Markdown-Formatierungen aus dem Lexikon-Eintrag bei (**fett**, *kursiv*, etc.)
- Verwende Markdown-Strukturierung wie **Überschriften**, *Hervorhebungen*, Listen
- Nutze Aufzählungen mit * oder - wenn sinnvoll
- Verwende **Fettschrift** für wichtige Begriffe und Konzepte
- Die Antwort soll gut strukturiert und mit Markdown formatiert sein

Antworte im folgenden JSON-Format:
{{
    "question": "Deine Frage hier",
    "answer": "Deine ausführliche, markdown-formatierte Antwort hier"
}}

Antworte NUR mit dem JSON, ohne zusätzliche Erklärungen.
"""
    return prompt

def submit_to_api(prompt, retries=3):
    """Sendet eine Anfrage an die gewählte API und holt die Antwort."""
    if USE_OPENAI_API:
        return submit_to_openai_api(prompt, retries)
    else:
        return submit_to_ollama_api(prompt, retries)

def submit_to_openai_api(prompt, retries=3):
    """Sendet eine Anfrage an die OpenAI-API und holt die Antwort."""
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": OPENAI_MODEL_NAME,
        "messages": [
            {
                "role": "system", 
                "content": "Du bist ein Experte für das Erstellen von Frage-Antwort-Paaren für Instruct-Datensätze. Antworte immer nur mit dem gewünschten JSON-Format und behalte alle Markdown-Formatierungen bei."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.4
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions", 
                json=payload, 
                headers=headers, 
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return content
            else:
                print(f"❌ OpenAI-API-Fehler {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            print(f"❌ API-Fehler (Versuch {attempt + 1}): {e}")
    
    return None

def submit_to_ollama_api(prompt, retries=3):
    """Sendet eine Anfrage an die Ollama-API und holt die Antwort."""
    headers = {
        'Authorization': f'Bearer {OLLAMA_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "temperature": 0.4,
        "stream": False,
        "messages": [
            {
                "role": "system", 
                "content": "Du bist ein Experte für das Erstellen von Frage-Antwort-Paaren für Instruct-Datensätze. Antworte immer nur mit dem gewünschten JSON-Format und behalte alle Markdown-Formatierungen bei."
            },
            {"role": "user", "content": prompt}
        ]
    }

    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "").strip()
                return content
            else:
                print(f"❌ Ollama-API-Fehler {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            print(f"❌ API-Fehler (Versuch {attempt + 1}): {e}")
    
    return None

def extract_qa_from_response(response):
    """Extrahiert QA-Paar aus der API-Antwort."""
    try:
        response = response.strip()
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        
        qa_data = json.loads(response)
        
        if "question" in qa_data and "answer" in qa_data:
            return {
                "question": qa_data["question"].strip(),
                "answer": qa_data["answer"].strip()
            }
        else:
            print(f"❌ Ungültiges QA-Format: {qa_data}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON-Parse-Fehler: {e}")
        print(f"Raw Response: {response}")
        return None

def generate_qa_for_entry(entry, num_questions=None):
    """Generiert 2-4 QA-Paare für einen Lexikon-Eintrag."""
    title = entry.get('title', '')
    lexikon_entry = entry.get('lexikon_entry', '')
    
    if not title or not lexikon_entry:
        print(f"⚠️ Unvollständiger Eintrag übersprungen: {title}")
        return []
    
    if num_questions is None:
        num_questions = random.randint(2, 4)
    
    print(f"🔄 Generiere {num_questions} QA-Paare für: {title}")
    
    qa_pairs = []
    used_question_types = set()
    
    for i in range(num_questions):
        available_types = [qt for qt in QUESTION_TYPES if qt["type"] not in used_question_types]
        if not available_types:
            available_types = QUESTION_TYPES
            used_question_types.clear()
        
        question_type = random.choice(available_types)
        used_question_types.add(question_type["type"])
        
        prompt = generate_qa_prompt(title, lexikon_entry, question_type)
        
        for attempt in range(3):
            response = submit_to_api(prompt)
            if response:
                qa_pair = extract_qa_from_response(response)
                if qa_pair:
                    qa_pair["source"] = entry.get('source', '')
                    qa_pair["title"] = title
                    qa_pair["question_type"] = question_type["type"]
                    qa_pairs.append(qa_pair)
                    print(f"   ✅ QA {i+1}/{num_questions} erstellt ({question_type['type']})")
                    break
            print(f"   ❌ Versuch {attempt + 1} fehlgeschlagen")
        
        if len(qa_pairs) <= i:
            print(f"   ⚠️ QA {i+1} konnte nicht erstellt werden")
    
    return qa_pairs

def convert_to_instruct_format(qa_pairs):
    """Konvertiert QA-Paare in das Standard-Instruct-Format."""
    instruct_dataset = []
    
    for qa in qa_pairs:
        instruct_entry = {
            "instruction": qa["question"],
            "input": "",
            "output": qa["answer"],
            "metadata": {
                "source": qa.get("source", ""),
                "title": qa.get("title", ""),
                "question_type": qa.get("question_type", "")
            }
        }
        instruct_dataset.append(instruct_entry)
    
    return instruct_dataset

def process_lexikon_to_qa_dataset(input_dir, output_dir, output_filename):
    """Hauptfunktion: Konvertiert alle Lexikon-Dateien zu einem QA-Datensatz."""
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🚀 Starte Konvertierung Lexikon → QA-Instruct-Datensatz")
    print(f"📂 Input-Verzeichnis: {input_dir}")
    print(f"📂 Output-Verzeichnis: {output_dir}")
    print(f"📄 Output-Datei: {output_filename}")
    print(f"🔧 API-Typ: {api_name}")
    print(f"🔧 Server: {server_url}")
    print(f"🔧 Modell: {model_name}")
    print(f"✨ Markdown-Formatierung: Aktiviert")
    
    if not check_api_connection():
        print("❌ API-Verbindung fehlgeschlagen. Verarbeitung abgebrochen.")
        return
    
    lexikon_entries = load_lexikon_files(input_dir)
    
    if not lexikon_entries:
        print("❌ Keine Lexikon-Einträge gefunden!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generiere QA-Paare für {len(lexikon_entries)} Lexikon-Einträge")
    print(f"{'='*60}")
    
    all_qa_pairs = []
    
    for idx, entry in enumerate(lexikon_entries, 1):
        title = entry.get('title', f'Eintrag {idx}')
        print(f"\n🔄 Eintrag {idx}/{len(lexikon_entries)}: {title}")
        
        qa_pairs = generate_qa_for_entry(entry)
        all_qa_pairs.extend(qa_pairs)
        
        print(f"   📊 {len(qa_pairs)} QA-Paare generiert")
    
    instruct_dataset = convert_to_instruct_format(all_qa_pairs)
    
    output_path = Path(output_dir) / output_filename
    
    final_dataset = {
        "metadata": {
            "total_entries": len(lexikon_entries),
            "total_qa_pairs": len(all_qa_pairs),
            "api_used": api_name,
            "model_used": model_name,
            "server_used": server_url,
            "format": "instruct",
            "markdown_formatting": True
        },
        "data": instruct_dataset
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"🎉 QA-Instruct-Datensatz erfolgreich erstellt!")
    print(f"📊 Statistiken:")
    print(f"   - API verwendet: {api_name}")
    print(f"   - Server: {server_url}")
    print(f"   - Modell: {model_name}")
    print(f"   - Lexikon-Einträge: {len(lexikon_entries)}")
    print(f"   - QA-Paare generiert: {len(all_qa_pairs)}")
    print(f"   - Durchschnitt: {len(all_qa_pairs)/len(lexikon_entries):.1f} QA pro Eintrag")
    print(f"   - Markdown-Formatierung: ✅ Aktiviert")
    print(f"   - Datei gespeichert: {output_path}")
    
    if instruct_dataset:
        print(f"\n📋 Beispiel-QA:")
        example = instruct_dataset[0]
        print(f"   Frage: {example['instruction']}")
        print(f"   Antwort: {example['output'][:100]}...")

if __name__ == "__main__":
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🔧 KONFIGURATION (aus zentraler Config):")
    print(f"   - API-Typ: {api_name}")
    print(f"   - Server: {server_url}")
    print(f"   - Modell: {model_name}")
    print(f"   - Markdown-Formatierung: ✅ Aktiviert")
    
    process_lexikon_to_qa_dataset(INPUT_DIR, OUTPUT_DIR, OUTPUT_FILENAME)