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
config_loader = PipelineConfigLoader("05_bmcreator")
config = config_loader.get_api_config()

# Extrahiere Konfigurationswerte
USE_OPENAI_API = config.get("use_openai_api", True)
OPENAI_BASE_URL = config.get("openai_base_url", "http://100.70.50.6:11434/v1")
OPENAI_API_KEY = config.get("openai_api_key", "ollama")
OPENAI_MODEL_NAME = config.get("openai_model_name", "gemma3:12b-it-qat")
OLLAMA_SERVER_URL = config.get("ollama_server_url", "http://100.70.50.6:11434")
OLLAMA_API_KEY = config.get("ollama_api_key", "ollama")
OLLAMA_MODEL_NAME = config.get("ollama_model_name", "gemma3:12b-it-qat")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_SERVER_URL}/api/chat"

# Verzeichnisse
INPUT_DIR = "INPUT"
OUTPUT_DIR = "BENCHMARKFRAGEN"
OUTPUT_FILENAME = "benchmark_fragen_complete.json"

# Kategorien-Mapping basierend auf Lexikon-Dateien
CATEGORY_MAPPING = {
    "CON01_Jahresabschluss": "Jahresabschlussanalyse",
    "CON02_Liquiditatsplanung": "Liquiditätsplanung", 
    "CON03_Budgetierung": "Budgetierung"
}

# Dynamische Fragenanzahl-Konfiguration aus Config
MAX_TOTAL_QUESTIONS = config.get("max_total_questions", 100)
MIN_QUESTIONS_PER_CATEGORY = config.get("min_questions_per_category", 5)
MAX_QUESTIONS_PER_CATEGORY = config.get("max_questions_per_category", 10)

# Fragetypen-Konfiguration aus Config
QUESTION_TYPE_DISTRIBUTION = config.get("question_type_distribution", {
    "definition": 0.7,
    "transfer": 0.3
})

# Zeige geladene Konfiguration
print("=" * 60)
print("📋 KONFIGURATION GELADEN (05_bmcreator)")
print("=" * 60)
config_loader.print_config_summary()
print(f"  📊 Max. Fragen gesamt: {MAX_TOTAL_QUESTIONS}")
print(f"  📊 Min. Fragen pro Kategorie: {MIN_QUESTIONS_PER_CATEGORY}")
print(f"  📊 Max. Fragen pro Kategorie: {MAX_QUESTIONS_PER_CATEGORY}")
print(f"  📊 Verteilung: {int(QUESTION_TYPE_DISTRIBUTION['definition']*100)}% Definition, {int(QUESTION_TYPE_DISTRIBUTION['transfer']*100)}% Transfer")
print("=" * 60)

# Frage-Templates für verschiedene Typen
DEFINITION_QUESTION_TEMPLATES = [
    "Was versteht man unter {title}?",
    "Wie wird {title} definiert?",
    "Erklären Sie den Begriff {title}.",
    "Was ist {title}?",
    "Definieren Sie {title}.",
    "Wie ermittelt man {title}?",
    "Was sind die Hauptmerkmale von {title}?",
    "Erläutern Sie {title}.",
    "Beschreiben Sie {title}.",
    "Welche Bedeutung hat {title}?",
    "Was charakterisiert {title}?",
    "Wie funktioniert {title}?",
    "Worin besteht {title}?",
    "Was umfasst {title}?",
    "Wie ist {title} aufgebaut?"
]

TRANSFER_QUESTION_TEMPLATES = [
    "Ein Unternehmen A möchte {title} implementieren. Wie sollte es vorgehen?",
    "Welche Schritte sind bei der Anwendung von {title} in einem mittelständischen Unternehmen zu beachten?",
    "Ein Controller muss {title} für sein Unternehmen bewerten. Worauf sollte er achten?",
    "Wie kann ein Unternehmen {title} in der Praxis umsetzen?",
    "Ein Finanzvorstand fragt Sie nach {title}. Wie erklären Sie ihm die praktische Relevanz?",
    "Welche Herausforderungen können bei der Umsetzung von {title} auftreten?",
    "Ein Start-up möchte {title} einführen. Welche Empfehlungen geben Sie?",
    "Wie wirkt sich {title} auf die Unternehmenspraxis aus?",
    "Ein Konzern plant die Optimierung von {title}. Welche Faktoren sind entscheidend?",
    "Welche praktischen Auswirkungen hat {title} auf das Controlling?",
    "Ein KMU hat Probleme mit {title}. Wie können diese gelöst werden?",
    "Warum ist {title} für Unternehmen wichtig und wie wird es angewendet?"
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
            return False
            
    except requests.RequestException as e:
        print(f"❌ OpenAI-API-Verbindung fehlgeschlagen: {e}")
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

def load_lexikon_files_by_category(input_dir):
    """Lädt alle Lexikon-JSON-Dateien und gruppiert sie nach Kategorien."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input-Verzeichnis '{input_dir}' existiert nicht!")
        return {}
    
    lexikon_files = list(input_path.glob("lexikon_*.json"))
    print(f"📁 Gefundene Lexikon-Dateien: {len(lexikon_files)}")
    
    categories = {}
    
    for file in lexikon_files:
        print(f"📖 Lade: {file.name}")
        
        # Bestimme Kategorie aus Dateiname
        category = "Sonstiges"  # Fallback
        for key, value in CATEGORY_MAPPING.items():
            if key in file.name:
                category = value
                break
        
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entries = data.get('lexikon_entries', [])
                print(f"   - {len(entries)} Einträge gefunden → Kategorie: {category}")
                
                if category not in categories:
                    categories[category] = []
                categories[category].extend(entries)
                
        except Exception as e:
            print(f"❌ Fehler beim Laden von {file.name}: {e}")
    
    # Zeige Kategorien-Übersicht
    print(f"\n📊 Kategorien-Übersicht:")
    for category, entries in categories.items():
        print(f"   - {category}: {len(entries)} Einträge")
    
    return categories

def select_question_type():
    """Wählt Fragetyp basierend auf Wahrscheinlichkeitsverteilung."""
    rand = random.random()
    if rand < QUESTION_TYPE_DISTRIBUTION["definition"]:
        return "definition"
    else:
        return "transfer"

def generate_question_template(title, question_type):
    """Generiert eine Frage basierend auf Typ und Titel."""
    if question_type == "definition":
        template = random.choice(DEFINITION_QUESTION_TEMPLATES)
    else:  # transfer
        template = random.choice(TRANSFER_QUESTION_TEMPLATES)
    
    return template.format(title=title)

def generate_benchmark_question_prompt(title, lexikon_entry, category, question_type=None):
    """Generiert einen Prompt für die Erstellung einer Benchmarkfrage."""
    
    # Wähle Fragetyp falls nicht vorgegeben
    if question_type is None:
        question_type = select_question_type()
    
    # Generiere Frage basierend auf Typ
    question_template = generate_question_template(title, question_type)
    
    if question_type == "definition":
        instruction = """
Erstelle eine präzise Definitionsfrage und beantworte sie fachlich korrekt.

WICHTIG - Antwort-Anforderungen:
- Verwende AUSSCHLIESSLICH Informationen aus dem bereitgestellten Lexikon-Eintrag
- Füge KEIN externes Wissen oder eigene Interpretation hinzu
- Verwende ALLE relevanten Informationen aus dem Lexikon-Eintrag
- Die Antwort muss vollumfänglich und vollständig sein
- Basiere deine Antwort zu 100% auf dem gegebenen Lexikon-Inhalt

Aufgabe:
1. Verwende die vorgegeben Frage (oder eine sehr ähnliche Variante)
2. Beantworte die Frage strukturiert und vollständig basierend NUR auf dem Lexikon-Eintrag
3. Die Antwort soll alle wichtigen Aspekte aus dem Lexikon-Eintrag abdecken
4. Fokus auf: Definition, Merkmale, Funktionsweise, Aufbau (alles aus dem Lexikon-Eintrag)
5. Stil: Fachlich, präzise, sachlich
6. Antwortlänge: So lang wie nötig um alle relevanten Informationen aus dem Lexikon-Eintrag abzudecken"""
    
    else:  # transfer
        instruction = """
Erstelle eine praxisorientierte Transferfrage und beantworte sie anwendungsbezogen.

WICHTIG - Antwort-Anforderungen:
- Verwende AUSSCHLIESSLICH Informationen aus dem bereitgestellten Lexikon-Eintrag
- Füge KEIN externes Wissen oder eigene Interpretation hinzu
- Verwende ALLE relevanten Informationen aus dem Lexikon-Eintrag
- Die Antwort muss vollumfänglich und vollständig sein
- Basiere deine Antwort zu 100% auf dem gegebenen Lexikon-Inhalt

Aufgabe:
1. Verwende die vorgegebene Frage (oder eine sehr ähnliche Variante)
2. Beantworte die Frage mit praktischem Bezug basierend NUR auf dem Lexikon-Eintrag
3. Die Antwort soll alle anwendungsrelevanten Aspekte aus dem Lexikon-Eintrag abdecken
4. Fokus auf: Praktische Bedeutung/Anwendung (alles aus dem Lexikon-Eintrag)
5. Stil: Praxisorientiert, beratend, umsetzungsfokussiert
6. Antwortlänge: So lang wie nötig um alle relevanten Informationen aus dem Lexikon-Eintrag abzudecken"""

    prompt = f"""
Du bist ein Experte für {category} und sollst eine Benchmarkfrage erstellen.

KRITISCH WICHTIG - Verwende als Antwort-Quelle AUSSCHLIESSLICH den folgenden Lexikon-Eintrag:
==========================================
Titel: {title}
Inhalt: {lexikon_entry}
==========================================

STRIKT BEFOLGEN:
- Verwende für die Antwort NUR Informationen aus dem obigen Lexikon-Eintrag
- Füge KEIN externes Wissen, keine eigenen Interpretationen oder Vermutungen hinzu
- Verwende ALLE relevanten Informationen aus dem Lexikon-Eintrag
- Die Antwort muss vollumfänglich auf dem Lexikon-Eintrag basieren
- Lass KEINE wichtigen Aspekte aus dem Lexikon-Eintrag weg

Vorgeschlagene Frage: {question_template}

{instruction}

Kategorien-Kontext: {category}
Fragetyp: {question_type}

Antworte im folgenden JSON-Format:
{{
    "frage": "Deine Benchmarkfrage hier (basierend auf dem Vorschlag)",
    "antwort": "Deine vollumfängliche Antwort hier - basierend NUR auf dem Lexikon-Eintrag"
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
                "content": "Du bist ein Experte für das Erstellen von Benchmarkfragen im Finanzcontrolling. Erstelle präzise, fachlich anspruchsvolle Fragen mit vollständigen Antworten. KRITISCH WICHTIG: Verwende für alle Antworten AUSSCHLIESSLICH die bereitgestellten Lexikon-Einträge als Wissensquelle. Füge NIEMALS externes Wissen hinzu."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.3
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions", 
                json=payload, 
                headers=headers, 
                timeout=90
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
        "temperature": 0.3,
        "stream": False,
        "messages": [
            {
                "role": "system", 
                "content": "Du bist ein Experte für das Erstellen von Benchmarkfragen im Finanzcontrolling. Erstelle präzise, fachlich anspruchsvolle Fragen mit vollständigen Antworten. KRITISCH WICHTIG: Verwende für alle Antworten AUSSCHLIESSLICH die bereitgestellten Lexikon-Einträge als Wissensquelle. Füge NIEMALS externes Wissen hinzu."
            },
            {"role": "user", "content": prompt}
        ]
    }

    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, headers=headers, timeout=90)
            
            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "").strip()
                return content
            else:
                print(f"❌ Ollama-API-Fehler {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            print(f"❌ API-Fehler (Versuch {attempt + 1}): {e}")
    
    return None

def extract_benchmark_qa_from_response(response):
    """Extrahiert Benchmarkfrage-Antwort-Paar aus der API-Antwort."""
    try:
        # Bereinige die Antwort
        response = response.strip()
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        
        qa_data = json.loads(response)
        
        # Validiere die Struktur
        if "frage" in qa_data and "antwort" in qa_data:
            return {
                "frage": qa_data["frage"].strip(),
                "antwort": qa_data["antwort"].strip()
            }
        else:
            print(f"❌ Ungültiges Benchmark-QA-Format: {list(qa_data.keys())}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON-Parse-Fehler: {e}")
        print(f"Raw Response (first 200 chars): {response[:200]}...")
        return None

def calculate_questions_per_category(categories_data):
    """Berechnet dynamisch die Anzahl Fragen pro Kategorie basierend auf verfügbaren Einträgen."""
    total_entries = sum(len(entries) for entries in categories_data.values())
    num_categories = len(categories_data)
    
    print(f"📊 Dynamische Fragenverteilung:")
    print(f"   - Kategorien gesamt: {num_categories}")
    print(f"   - Einträge gesamt: {total_entries}")
    print(f"   - Maximum Fragen: {MAX_TOTAL_QUESTIONS}")
    
    questions_distribution = {}
    
    # Berechne Fragen pro Kategorie basierend auf Einträgen
    for category, entries in categories_data.items():
        num_entries = len(entries)
        
        if num_entries <= 10:
            # Kleine Kategorien: 5-10 Fragen
            num_questions = min(num_entries, random.randint(MIN_QUESTIONS_PER_CATEGORY, MAX_QUESTIONS_PER_CATEGORY))
        elif num_entries <= 30:
            # Mittlere Kategorien: 10-20 Fragen
            num_questions = min(num_entries, random.randint(10, 20))
        else:
            # Große Kategorien: 15-35 Fragen
            num_questions = min(num_entries, random.randint(15, 35))
        
        questions_distribution[category] = num_questions
        print(f"   - {category}: {num_questions} Fragen (aus {num_entries} Einträgen)")
    
    # Überprüfe Gesamtanzahl und passe an falls nötig
    total_planned = sum(questions_distribution.values())
    
    if total_planned > MAX_TOTAL_QUESTIONS:
        print(f"   ⚠️ Geplant: {total_planned} Fragen → Reduzierung auf {MAX_TOTAL_QUESTIONS}")
        
        # Proportionale Reduzierung
        reduction_factor = MAX_TOTAL_QUESTIONS / total_planned
        
        for category in questions_distribution:
            original = questions_distribution[category]
            questions_distribution[category] = max(1, int(original * reduction_factor))
        
        # Feinabstimmung falls noch Fragen übrig sind
        current_total = sum(questions_distribution.values())
        remaining = MAX_TOTAL_QUESTIONS - current_total
        
        categories_list = list(questions_distribution.keys())
        for i in range(remaining):
            category = categories_list[i % len(categories_list)]
            if questions_distribution[category] < len(categories_data[category]):
                questions_distribution[category] += 1
    
    final_total = sum(questions_distribution.values())
    print(f"   ✅ Final: {final_total} Fragen verteilt")
    
    return questions_distribution

def generate_benchmark_questions_for_category(category, entries, num_questions):
    """Generiert Benchmarkfragen für eine Kategorie."""
    print(f"\n🔄 Generiere {num_questions} Benchmarkfragen für Kategorie: {category}")
    print(f"   📊 Verfügbare Einträge: {len(entries)}")
    
    # Berechne Anzahl Definition vs. Transfer Fragen
    num_definition = int(num_questions * QUESTION_TYPE_DISTRIBUTION["definition"])
    num_transfer = num_questions - num_definition
    
    print(f"   🎯 Fragetyp-Verteilung: {num_definition} Definitionen, {num_transfer} Transfer")
    
    if len(entries) < num_questions:
        print(f"   ⚠️ Nur {len(entries)} Einträge verfügbar, generiere {len(entries)} Fragen")
        num_questions = len(entries)
    
    # Zufällige Auswahl der Einträge (WICHTIG: Nicht die ersten!)
    selected_entries = random.sample(entries, num_questions)
    print(f"   🎲 {num_questions} Einträge zufällig ausgewählt")
    
    # Erstelle Liste der Fragetypen
    question_types = (["definition"] * num_definition + ["transfer"] * num_transfer)
    random.shuffle(question_types)  # Mische die Reihenfolge
    
    benchmark_questions = []
    definition_count = 0
    transfer_count = 0
    
    for idx, entry in enumerate(selected_entries, 1):
        title = entry.get('title', f'Eintrag {idx}')
        lexikon_entry = entry.get('lexikon_entry', '')
        
        if not title or not lexikon_entry:
            print(f"   ⚠️ Unvollständiger Eintrag übersprungen: {title}")
            continue
        
        # Bestimme Fragetyp für diese Frage
        question_type = question_types[idx-1] if idx-1 < len(question_types) else select_question_type()
        
        print(f"   🔄 Frage {idx}/{num_questions}: {title} ({question_type})")
        
        # Generiere Benchmarkfrage
        prompt = generate_benchmark_question_prompt(title, lexikon_entry, category, question_type)
        
        for attempt in range(3):
            response = submit_to_api(prompt)
            if response:
                qa_pair = extract_benchmark_qa_from_response(response)
                if qa_pair:
                    # Bestimme ID basierend auf Kategorie
                    category_prefix = {
                        "Jahresabschlussanalyse": "JA",
                        "Liquiditätsplanung": "LP", 
                        "Budgetierung": "BU"
                    }.get(category, "XX")
                    
                    benchmark_question = {
                        "id": f"{category_prefix}_{idx:02d}",
                        "frage": qa_pair["frage"],
                        "antwort": qa_pair["antwort"]
                    }
                    
                    benchmark_questions.append(benchmark_question)
                    
                    # Zähle Fragetypen
                    if question_type == "definition":
                        definition_count += 1
                    else:
                        transfer_count += 1
                    
                    print(f"      ✅ {question_type.title()}-Frage erstellt")
                    break
            print(f"      ❌ Versuch {attempt + 1} fehlgeschlagen")
        
        if len(benchmark_questions) < idx:
            print(f"      ⚠️ Benchmarkfrage {idx} konnte nicht erstellt werden")
    
    print(f"   📊 {len(benchmark_questions)} Benchmarkfragen erfolgreich generiert")
    print(f"   📋 Tatsächliche Verteilung: {definition_count} Definitionen, {transfer_count} Transfer")
    return benchmark_questions

def create_benchmark_dataset(categories_data):
    """Erstellt den kompletten Benchmark-Datensatz im gewünschten Format."""
    
    dataset = {
        "titel": "Benchmark-Fragen Finanzcontrolling",
        "beschreibung": "Sammlung von Benchmark-Fragen für das Finanzcontrolling in den Bereichen Jahresabschlussanalyse, Budgetierung und Liquiditätsplanung",
        "kategorien": []
    }
    
    # Berechne dynamische Fragenverteilung
    questions_distribution = calculate_questions_per_category(categories_data)
    
    total_questions = 0
    
    for category, entries in categories_data.items():
        if not entries or category not in questions_distribution:
            continue
            
        print(f"🏷️ Verarbeite Kategorie: {category}")
        
        num_questions = questions_distribution[category]
        benchmark_questions = generate_benchmark_questions_for_category(
            category, entries, num_questions
        )
        
        if benchmark_questions:
            category_data = {
                "kategorie": category,
                "anzahl_fragen": len(benchmark_questions),
                "fragen": benchmark_questions
            }
            
            dataset["kategorien"].append(category_data)
            total_questions += len(benchmark_questions)
    
    print(f"\n📊 Benchmark-Datensatz erstellt:")
    print(f"   - Kategorien: {len(dataset['kategorien'])}")
    print(f"   - Gesamt-Fragen: {total_questions}")
    
    return dataset

def main():
    """Hauptfunktion: Erstellt Benchmark-Fragen aus Lexikon-Einträgen."""
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🚀 Starte Benchmark-Fragen-Generator")
    print(f"📂 Input-Verzeichnis: {INPUT_DIR} (alle lexikon_*.json Dateien)")
    print(f"📂 Output-Verzeichnis: {OUTPUT_DIR}")
    print(f"📄 Output-Datei: {OUTPUT_FILENAME}")
    print(f"🔧 API-Typ: {api_name}")
    print(f"🔧 Server: {server_url}")
    print(f"🔧 Modell: {model_name}")
    print(f"🎯 Max. Fragen gesamt: {MAX_TOTAL_QUESTIONS}")
    print(f"🎯 Kleine Kategorien: {MIN_QUESTIONS_PER_CATEGORY}-{MAX_QUESTIONS_PER_CATEGORY} Fragen")
    print(f"🎯 Fragetyp-Verteilung: {int(QUESTION_TYPE_DISTRIBUTION['definition']*100)}% Definitionen, {int(QUESTION_TYPE_DISTRIBUTION['transfer']*100)}% Transfer")
    
    # Teste API-Verbindung
    if not check_api_connection():
        print("❌ API-Verbindung fehlgeschlagen. Verarbeitung abgebrochen.")
        return
    
    # Lade Lexikon-Einträge nach Kategorien
    categories_data = load_lexikon_files_by_category(INPUT_DIR)
    
    if not categories_data:
        print("❌ Keine Lexikon-Einträge gefunden!")
        return
    
    # Erstelle Output-Verzeichnis
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generiere Benchmark-Fragen")
    print(f"{'='*60}")
    
    # Erstelle Benchmark-Datensatz
    benchmark_dataset = create_benchmark_dataset(categories_data)
    
    # Speichere den Datensatz
    output_path = Path(OUTPUT_DIR) / OUTPUT_FILENAME
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"🎉 Benchmark-Fragen erfolgreich erstellt!")
    print(f"📊 Statistiken:")
    print(f"   - API verwendet: {api_name}")
    print(f"   - Server: {server_url}")
    print(f"   - Modell: {model_name}")
    
    total_questions = sum(cat["anzahl_fragen"] for cat in benchmark_dataset["kategorien"])
    total_entries = sum(len(entries) for entries in categories_data.values())
    
    print(f"   - Kategorien: {len(benchmark_dataset['kategorien'])}")
    print(f"   - Benchmark-Fragen: {total_questions} (max. {MAX_TOTAL_QUESTIONS})")
    print(f"   - Verfügbare Einträge: {total_entries}")
    print(f"   - Alle lexikon_*.json Dateien verarbeitet: ✅")
    print(f"   - Datei gespeichert: {output_path}")
    
    # Zeige Beispiel-Fragen
    print(f"\n📋 Beispiel-Benchmarkfragen:")
    for category_data in benchmark_dataset["kategorien"][:2]:  # Erste 2 Kategorien
        if category_data["fragen"]:
            example = category_data["fragen"][0]
            print(f"   [{category_data['kategorie']}] {example['id']}")
            print(f"   Frage: {example['frage']}")
            print(f"   Antwort: {example['antwort'][:100]}...")
            print()

if __name__ == "__main__":
    # Zeige Konfiguration
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🔧 KONFIGURATION (aus zentraler Config):")
    print(f"   - API-Typ: {api_name}")
    print(f"   - Server: {server_url}")
    print(f"   - Modell: {model_name}")
    print(f"   - Zufällige Auswahl: ✅ Aktiviert")
    
    # Hauptausführung
    main()