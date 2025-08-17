import json
import requests
import random
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

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

# ========================================
# ERWEITERTE FRAGETYPEN FÜR BESSERE ABDECKUNG
# ========================================

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
    },
    {
        "type": "technical_specs",
        "templates": [
            "Welche technischen Daten hat {title}?",
            "Nennen Sie alle technischen Spezifikationen von {title}.",
            "Welche Kennzahlen und Parameter hat {title}?",
            "Listen Sie die technischen Details von {title} auf.",
            "Was sind die technischen Eigenschaften von {title}?"
        ]
    },
    {
        "type": "complete_overview",
        "templates": [
            "Geben Sie eine vollständige Übersicht über {title}.",
            "Fassen Sie alle wichtigen Informationen zu {title} zusammen.",
            "Was sind alle relevanten Details zu {title}?",
            "Beschreiben Sie {title} mit allen verfügbaren Informationen.",
            "Erklären Sie {title} umfassend mit allen Details."
        ]
    },
    {
        "type": "data_and_facts",
        "templates": [
            "Welche konkreten Daten und Fakten gibt es zu {title}?",
            "Nennen Sie alle Zahlen und Messwerte zu {title}.",
            "Welche quantitativen Informationen liegen zu {title} vor?",
            "Listen Sie alle faktischen Angaben zu {title} auf.",
            "Was sind die messbaren Eigenschaften von {title}?"
        ]
    },
    {
        "type": "structure_and_components",
        "templates": [
            "Aus welchen Komponenten besteht {title}?",
            "Wie ist {title} strukturiert?",
            "Welche Bestandteile hat {title}?",
            "Beschreiben Sie den Aufbau von {title}.",
            "Welche Elemente gehören zu {title}?"
        ]
    },
    {
        "type": "comparison_and_differences",
        "templates": [
            "Wie unterscheidet sich {title} von ähnlichen Konzepten?",
            "Was macht {title} besonders?",
            "Welche Varianten von {title} gibt es?",
            "Wie grenzt sich {title} ab?",
            "Was sind die Alleinstellungsmerkmale von {title}?"
        ]
    }
]

# ========================================
# API-VERBINDUNG
# ========================================

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

# ========================================
# DATEI-VERARBEITUNG
# ========================================

def load_lexikon_files(input_dir: str) -> List[Dict[str, Any]]:
    """Lädt alle Lexikon-JSON-Dateien aus dem Input-Verzeichnis."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input-Verzeichnis '{input_dir}' existiert nicht!")
        return []
    
    # Suche nach allen möglichen Lexikon-Dateien
    lexikon_patterns = ["lexikon_*.json", "processed_*.json", "*_lexikon.json"]
    lexikon_files = []
    
    for pattern in lexikon_patterns:
        lexikon_files.extend(list(input_path.glob(pattern)))
    
    # Entferne Duplikate
    lexikon_files = list(set(lexikon_files))
    
    print(f"📁 Gefundene Lexikon-Dateien: {len(lexikon_files)}")
    
    all_entries = []
    
    for file in lexikon_files:
        print(f"📖 Lade: {file.name}")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Unterstütze verschiedene Strukturen
                if 'lexikon_entries' in data:
                    entries = data.get('lexikon_entries', [])
                elif 'entries' in data:
                    entries = data.get('entries', [])
                elif 'data' in data:
                    entries = data.get('data', [])
                else:
                    # Fallback: Versuche die Datei als Liste zu interpretieren
                    if isinstance(data, list):
                        entries = data
                    else:
                        entries = []
                
                print(f"   - {len(entries)} Einträge gefunden")
                all_entries.extend(entries)
                
        except Exception as e:
            print(f"❌ Fehler beim Laden von {file.name}: {e}")
    
    print(f"📊 Gesamt: {len(all_entries)} Lexikon-Einträge")
    return all_entries

# ========================================
# PROMPT-GENERIERUNG MIT FAKTEN-ERHALTUNG
# ========================================

def detect_content_features(lexikon_entry: str) -> Dict[str, bool]:
    """Erkennt spezielle Inhaltsmerkmale im Lexikon-Eintrag."""
    features = {
        'has_table': '|' in lexikon_entry or '\t' in lexikon_entry,
        'has_technical_data': any(keyword in lexikon_entry.lower() for keyword in 
                                 ['temperaturbereich', 'messbereich', 'spezifikationen', 
                                  'betriebsspannung', 'frequenz', 'genauigkeit']),
        'has_numbers': any(char.isdigit() for char in lexikon_entry),
        'has_list': any(marker in lexikon_entry for marker in ['•', '●', '○', '■', '- ', '* ', '1.', '2.']),
        'has_product_code': bool(re.search(r'\b[A-Z]{2,4}-\d{2,4}\b', lexikon_entry)) if 're' in globals() else False,
        'is_structured_data': 'bundesland' in lexikon_entry.lower() or 'tabelle' in lexikon_entry.lower()
    }
    return features

def generate_qa_prompt(title: str, lexikon_entry: str, question_type: Dict[str, Any]) -> str:
    """Generiert einen Prompt für die QA-Erstellung mit vollständiger Fakten-Erhaltung."""
    
    # Erkenne Content-Features
    features = detect_content_features(lexikon_entry)
    
    # Baue spezielle Instruktionen basierend auf Content-Features
    special_instructions = []
    
    if features['has_table']:
        special_instructions.append("""
- Der Lexikon-Eintrag enthält TABELLEN oder strukturierte Daten - diese müssen VOLLSTÄNDIG übernommen werden
- Verwende Markdown-Tabellen (|---|---|) für tabellarische Daten
- Behalte die Struktur und Formatierung bei""")
    
    if features['has_technical_data']:
        special_instructions.append("""
- Der Eintrag enthält TECHNISCHE DATEN - alle Spezifikationen, Werte und Parameter müssen erhalten bleiben
- Verwende **Fettschrift** für technische Begriffe
- Strukturiere technische Daten als Liste oder Tabelle""")
    
    if features['has_numbers']:
        special_instructions.append("""
- ALLE Zahlen, Messwerte, Prozentangaben und quantitative Daten müssen exakt übernommen werden
- Runde keine Werte und ändere keine Einheiten""")
    
    if features['has_list']:
        special_instructions.append("""
- Behalte alle Aufzählungen und Listen bei
- Verwende die gleiche Listenstruktur wie im Original""")
    
    if features['has_product_code']:
        special_instructions.append("""
- Dies ist ein PRODUKTEINTRAG - behalte Produktcodes und -namen exakt bei
- Behandle es als konkretes Produkt, nicht als allgemeines Konzept""")
    
    special_instructions_text = "\n".join(special_instructions) if special_instructions else ""
    
    # Wähle einen zufälligen Template für die Frage
    question_template = random.choice(question_type["templates"])
    
    prompt = f"""
Erstelle ein hochwertiges Frage-Antwort-Paar für einen Instruct-Datensatz.

THEMA: {title}
FRAGETYP: {question_type["type"]}
FRAGE-TEMPLATE: {question_template}

LEXIKON-EINTRAG (QUELLE):
{lexikon_entry}

AUFGABE:
1. Erstelle eine natürliche Frage basierend auf dem Template "{question_template}"
2. Beantworte die Frage VOLLSTÄNDIG und AUSFÜHRLICH basierend auf dem Lexikon-Eintrag
3. Die Antwort soll umfassend sein (3-10 Sätze oder mehr bei komplexen Themen)
4. Übernehme ALLE relevanten Informationen, Fakten und Details aus dem Lexikon-Eintrag
5. Stil: Professionell, lehrreich, wie ein Fachexperte
6. Sprache: Deutsch

KRITISCHE ANFORDERUNGEN:
- KEIN Informationsverlust! Jedes Detail, jede Zahl, jeder Fakt muss erhalten bleiben
- Bei technischen Themen: ALLE Spezifikationen und Werte übernehmen
- Bei Produkten: Produktcodes und -namen exakt beibehalten
- Bei Listen/Tabellen: Struktur vollständig übernehmen

MARKDOWN-FORMATIERUNG:
- Verwende **Fettschrift** für wichtige Begriffe und Überschriften
- Verwende *Kursivschrift* für Hervorhebungen
- Strukturiere mit Überschriften (##, ###) wenn sinnvoll
- Nutze Listen (- oder *) für Aufzählungen
- Verwende Markdown-Tabellen für tabellarische Daten
- Code-Blöcke ``` für technische Werte wenn passend

SPEZIELLE CONTENT-ANWEISUNGEN:
{special_instructions_text}

ANTWORTLÄNGE:
- Minimum: 3 vollständige Sätze
- Maximum: So lang wie nötig um ALLE Informationen zu vermitteln
- Bei komplexen Themen oder vielen Daten: Gerne auch 15-20 Sätze

FORMAT DER AUSGABE:
{{
    "question": "Die formulierte Frage basierend auf dem Template",
    "answer": "Die vollständige, ausführliche, faktenreiche Antwort mit allen Details aus dem Lexikon-Eintrag"
}}

WICHTIG: Antworte NUR mit dem JSON-Objekt, keine zusätzlichen Erklärungen!
"""
    return prompt

# ========================================
# API-KOMMUNIKATION
# ========================================

def submit_to_api(prompt: str, retries: int = 3) -> Optional[str]:
    """Sendet eine Anfrage an die gewählte API und holt die Antwort."""
    if USE_OPENAI_API:
        return submit_to_openai_api(prompt, retries)
    else:
        return submit_to_ollama_api(prompt, retries)

def submit_to_openai_api(prompt: str, retries: int = 3) -> Optional[str]:
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
                "content": "Du bist ein Experte für das Erstellen von hochwertigen Frage-Antwort-Paaren für Instruct-Datensätze. Deine Antworten sind vollständig, faktenreich und behalten ALLE Details aus der Quelle bei. Du verwendest Markdown-Formatierung für bessere Struktur."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,  # Erhöht für längere, vollständige Antworten
        "temperature": 0.3   # Niedrig für faktentreue
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

def submit_to_ollama_api(prompt: str, retries: int = 3) -> Optional[str]:
    """Sendet eine Anfrage an die Ollama-API und holt die Antwort."""
    headers = {
        'Authorization': f'Bearer {OLLAMA_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "temperature": 0.3,  # Niedrig für faktentreue
        "stream": False,
        "messages": [
            {
                "role": "system", 
                "content": "Du bist ein Experte für das Erstellen von hochwertigen Frage-Antwort-Paaren für Instruct-Datensätze. Deine Antworten sind vollständig, faktenreich und behalten ALLE Details aus der Quelle bei. Du verwendest Markdown-Formatierung für bessere Struktur."
            },
            {"role": "user", "content": prompt}
        ],
        "options": {
            "num_predict": 2000  # Erhöht für längere Antworten
        }
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

# ========================================
# QA-GENERIERUNG
# ========================================

def extract_qa_from_response(response: str) -> Optional[Dict[str, str]]:
    """Extrahiert QA-Paar aus der API-Antwort."""
    try:
        # Bereinige die Antwort
        response = response.strip()
        
        # Entferne Code-Block-Marker falls vorhanden
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        
        # Parse JSON
        qa_data = json.loads(response)
        
        # Validiere Struktur
        if "question" in qa_data and "answer" in qa_data:
            return {
                "question": qa_data["question"].strip(),
                "answer": qa_data["answer"].strip()
            }
        else:
            print(f"❌ Ungültiges QA-Format: Fehlende Felder")
            return None
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON-Parse-Fehler: {e}")
        print(f"Raw Response (erste 200 Zeichen): {response[:200]}...")
        return None
    except Exception as e:
        print(f"❌ Unerwarteter Fehler beim Extrahieren: {e}")
        return None

def generate_qa_for_entry(entry: Dict[str, Any], num_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generiert mehrere QA-Paare für einen Lexikon-Eintrag mit Fokus auf Vollständigkeit."""
    
    # Extrahiere relevante Felder (unterstütze verschiedene Strukturen)
    title = entry.get('title', entry.get('name', ''))
    lexikon_entry = entry.get('lexikon_entry', entry.get('entry', entry.get('content', '')))
    
    if not title or not lexikon_entry:
        print(f"⚠️ Unvollständiger Eintrag übersprungen: {title or 'Unbekannt'}")
        return []
    
    # Bestimme Anzahl der Fragen basierend auf Content-Länge
    if num_questions is None:
        content_length = len(lexikon_entry)
        if content_length < 500:
            num_questions = random.randint(3, 5)
        elif content_length < 1500:
            num_questions = random.randint(4, 7)
        else:
            num_questions = random.randint(5, 8)
    
    print(f"🔄 Generiere {num_questions} QA-Paare für: {title}")
    
    qa_pairs = []
    used_question_types = set()
    
    # GARANTIERE eine "complete_overview" Frage als erste für maximale Abdeckung
    overview_type = next((qt for qt in QUESTION_TYPES if qt["type"] == "complete_overview"), None)
    
    if overview_type:
        print(f"   🎯 Erstelle vollständige Übersicht...")
        prompt = generate_qa_prompt(title, lexikon_entry, overview_type)
        
        for attempt in range(3):
            response = submit_to_api(prompt)
            if response:
                qa_pair = extract_qa_from_response(response)
                if qa_pair:
                    qa_pair["source"] = entry.get('source', '')
                    qa_pair["title"] = title
                    qa_pair["question_type"] = "complete_overview"
                    qa_pairs.append(qa_pair)
                    used_question_types.add("complete_overview")
                    print(f"   ✅ QA 1/{num_questions} erstellt (complete_overview)")
                    break
            if attempt < 2:
                print(f"   ⚠️ Versuch {attempt + 1} fehlgeschlagen, wiederhole...")
    
    # Priorisiere bestimmte Fragetypen basierend auf Content
    features = detect_content_features(lexikon_entry)
    priority_types = []
    
    if features['has_technical_data']:
        priority_types.append("technical_specs")
    if features['has_numbers']:
        priority_types.append("data_and_facts")
    if features['has_list'] or features['has_table']:
        priority_types.append("structure_and_components")
    
    # Füge priorisierte Fragen hinzu
    for priority_type in priority_types:
        if len(qa_pairs) >= num_questions:
            break
            
        question_type = next((qt for qt in QUESTION_TYPES if qt["type"] == priority_type), None)
        if question_type and priority_type not in used_question_types:
            prompt = generate_qa_prompt(title, lexikon_entry, question_type)
            
            for attempt in range(2):
                response = submit_to_api(prompt)
                if response:
                    qa_pair = extract_qa_from_response(response)
                    if qa_pair:
                        qa_pair["source"] = entry.get('source', '')
                        qa_pair["title"] = title
                        qa_pair["question_type"] = priority_type
                        qa_pairs.append(qa_pair)
                        used_question_types.add(priority_type)
                        print(f"   ✅ QA {len(qa_pairs)}/{num_questions} erstellt ({priority_type})")
                        break
    
    # Fülle mit weiteren zufälligen Fragetypen auf
    remaining_questions = num_questions - len(qa_pairs)
    
    for i in range(remaining_questions):
        # Wähle unbenutzte Fragetypen
        available_types = [qt for qt in QUESTION_TYPES if qt["type"] not in used_question_types]
        
        # Wenn alle verwendet wurden, erlaube Wiederholung
        if not available_types:
            available_types = QUESTION_TYPES
            used_question_types.clear()
        
        question_type = random.choice(available_types)
        used_question_types.add(question_type["type"])
        
        prompt = generate_qa_prompt(title, lexikon_entry, question_type)
        
        for attempt in range(2):
            response = submit_to_api(prompt)
            if response:
                qa_pair = extract_qa_from_response(response)
                if qa_pair:
                    qa_pair["source"] = entry.get('source', '')
                    qa_pair["title"] = title
                    qa_pair["question_type"] = question_type["type"]
                    qa_pairs.append(qa_pair)
                    print(f"   ✅ QA {len(qa_pairs)}/{num_questions} erstellt ({question_type['type']})")
                    break
        
        if len(qa_pairs) <= len(qa_pairs):
            print(f"   ⚠️ QA {len(qa_pairs)+1} konnte nicht erstellt werden")
    
    return qa_pairs

# ========================================
# DATENSATZ-KONVERTIERUNG
# ========================================

def convert_to_instruct_format(qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Konvertiert QA-Paare in das Standard-Instruct-Format."""
    instruct_dataset = []
    
    for qa in qa_pairs:
        instruct_entry = {
            "instruction": qa["question"],
            "input": "",  # Leer für diesen Use-Case
            "output": qa["answer"],
            "metadata": {
                "source": qa.get("source", ""),
                "title": qa.get("title", ""),
                "question_type": qa.get("question_type", ""),
                "answer_length": len(qa["answer"]),
                "has_markdown": "**" in qa["answer"] or "*" in qa["answer"] or "#" in qa["answer"]
            }
        }
        instruct_dataset.append(instruct_entry)
    
    return instruct_dataset

# ========================================
# HAUPTVERARBEITUNG
# ========================================

def process_lexikon_to_qa_dataset(input_dir: str, output_dir: str, output_filename: str):
    """Hauptfunktion: Konvertiert alle Lexikon-Dateien zu einem umfassenden QA-Datensatz."""
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🚀 Starte erweiterte Konvertierung Lexikon → QA-Instruct-Datensatz")
    print(f"📂 Input-Verzeichnis: {input_dir}")
    print(f"📂 Output-Verzeichnis: {output_dir}")
    print(f"📄 Output-Datei: {output_filename}")
    print(f"🔧 API-Typ: {api_name}")
    print(f"🔧 Server: {server_url}")
    print(f"🔧 Modell: {model_name}")
    print(f"✨ Features:")
    print(f"   - Erweiterte Fragetypen: {len(QUESTION_TYPES)} Kategorien")
    print(f"   - Vollständige Fakten-Erhaltung: ✅")
    print(f"   - Markdown-Formatierung: ✅")
    print(f"   - Dynamische Fragenzahl: ✅")
    print(f"   - Tabellen-Unterstützung: ✅")
    
    # API-Verbindung prüfen
    if not check_api_connection():
        print("❌ API-Verbindung fehlgeschlagen. Verarbeitung abgebrochen.")
        return
    
    # Lade Lexikon-Einträge
    lexikon_entries = load_lexikon_files(input_dir)
    
    if not lexikon_entries:
        print("❌ Keine Lexikon-Einträge gefunden!")
        return
    
    # Output-Verzeichnis erstellen
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generiere QA-Paare für {len(lexikon_entries)} Lexikon-Einträge")
    print(f"{'='*60}")
    
    all_qa_pairs = []
    statistics = {
        'total_entries': len(lexikon_entries),
        'processed_entries': 0,
        'failed_entries': 0,
        'total_qa_pairs': 0,
        'qa_by_type': {}
    }
    
    # Verarbeite jeden Eintrag
    for idx, entry in enumerate(lexikon_entries, 1):
        title = entry.get('title', entry.get('name', f'Eintrag {idx}'))
        print(f"\n🔄 Eintrag {idx}/{len(lexikon_entries)}: {title}")
        
        try:
            qa_pairs = generate_qa_for_entry(entry)
            
            if qa_pairs:
                all_qa_pairs.extend(qa_pairs)
                statistics['processed_entries'] += 1
                statistics['total_qa_pairs'] += len(qa_pairs)
                
                # Statistik nach Typ
                for qa in qa_pairs:
                    q_type = qa.get('question_type', 'unknown')
                    statistics['qa_by_type'][q_type] = statistics['qa_by_type'].get(q_type, 0) + 1
                
                print(f"   📊 {len(qa_pairs)} QA-Paare generiert")
            else:
                statistics['failed_entries'] += 1
                print(f"   ⚠️ Keine QA-Paare generiert")
                
        except Exception as e:
            statistics['failed_entries'] += 1
            print(f"   ❌ Fehler bei Verarbeitung: {e}")
    
    # Konvertiere zu Instruct-Format
    instruct_dataset = convert_to_instruct_format(all_qa_pairs)
    
    # Berechne zusätzliche Statistiken
    avg_answer_length = sum(entry['metadata']['answer_length'] for entry in instruct_dataset) / len(instruct_dataset) if instruct_dataset else 0
    markdown_count = sum(1 for entry in instruct_dataset if entry['metadata']['has_markdown'])
    
    # Erstelle finalen Datensatz
    output_path = Path(output_dir) / output_filename
    
    final_dataset = {
        "metadata": {
            "total_entries": statistics['total_entries'],
            "processed_entries": statistics['processed_entries'],
            "failed_entries": statistics['failed_entries'],
            "total_qa_pairs": statistics['total_qa_pairs'],
            "qa_by_type": statistics['qa_by_type'],
            "avg_qa_per_entry": statistics['total_qa_pairs'] / statistics['processed_entries'] if statistics['processed_entries'] > 0 else 0,
            "avg_answer_length": avg_answer_length,
            "markdown_formatted_answers": markdown_count,
            "api_used": api_name,
            "model_used": model_name,
            "server_used": server_url,
            "format": "instruct",
            "features": {
                "markdown_formatting": True,
                "complete_overview_guaranteed": True,
                "fact_preservation": True,
                "table_support": True,
                "extended_question_types": len(QUESTION_TYPES)
            }
        },
        "data": instruct_dataset
    }
    
    # Speichere Datensatz
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    
    # Ausgabe der Statistiken
    print(f"\n{'='*60}")
    print(f"🎉 QA-Instruct-Datensatz erfolgreich erstellt!")
    print(f"{'='*60}")
    print(f"📊 Detaillierte Statistiken:")
    print(f"   📁 Verarbeitung:")
    print(f"      - Lexikon-Einträge gesamt: {statistics['total_entries']}")
    print(f"      - Erfolgreich verarbeitet: {statistics['processed_entries']}")
    print(f"      - Fehlgeschlagen: {statistics['failed_entries']}")
    print(f"   📝 QA-Paare:")
    print(f"      - Gesamt generiert: {statistics['total_qa_pairs']}")
    print(f"      - Durchschnitt pro Eintrag: {statistics['total_qa_pairs']/statistics['processed_entries']:.1f}" if statistics['processed_entries'] > 0 else "")
    print(f"      - Durchschnittliche Antwortlänge: {avg_answer_length:.0f} Zeichen")
    print(f"      - Mit Markdown-Formatierung: {markdown_count}/{len(instruct_dataset)}")
    print(f"   🎯 Fragetypen:")
    for q_type, count in sorted(statistics['qa_by_type'].items(), key=lambda x: x[1], reverse=True):
        print(f"      - {q_type}: {count}")
    print(f"   🔧 Technische Details:")
    print(f"      - API: {api_name}")
    print(f"      - Server: {server_url}")
    print(f"      - Modell: {model_name}")
    print(f"   💾 Ausgabe:")
    print(f"      - Datei: {output_path}")
    print(f"      - Größe: {output_path.stat().st_size / 1024:.1f} KB" if output_path.exists() else "")
    
    # Zeige Beispiele
    if instruct_dataset:
        print(f"\n📋 Beispiel-QA-Paare:")
        
        # Zeige verschiedene Fragetypen
        shown_types = set()
        examples_shown = 0
        
        for entry in instruct_dataset:
            q_type = entry['metadata']['question_type']
            if q_type not in shown_types and examples_shown < 3:
                print(f"\n   Typ: {q_type}")
                print(f"   Frage: {entry['instruction']}")
                answer_preview = entry['output'][:200] + "..." if len(entry['output']) > 200 else entry['output']
                print(f"   Antwort: {answer_preview}")
                shown_types.add(q_type)
                examples_shown += 1

# ========================================
# HAUPTPROGRAMM
# ========================================

if __name__ == "__main__":
    # Importiere regex für erweiterte Pattern-Matching
    import re
    
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🔧 KONFIGURATION (aus zentraler Config):")
    print(f"   - API-Typ: {api_name}")
    print(f"   - Server: {server_url}")
    print(f"   - Modell: {model_name}")
    print(f"   - Erweiterte Features: ✅")
    print(f"   - Fakten-Erhaltung: ✅")
    print(f"   - Markdown-Formatierung: ✅")
    print(f"   - Tabellen-Unterstützung: ✅")
    
    # Starte Verarbeitung
    process_lexikon_to_qa_dataset(INPUT_DIR, OUTPUT_DIR, OUTPUT_FILENAME)