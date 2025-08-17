import json
import requests
import re
import os
import sys
from pathlib import Path

# ========================================
# ZENTRALE KONFIGURATION LADEN
# ========================================
sys.path.append(str(Path(__file__).parent.parent.parent))  # Zum Hauptverzeichnis
from config_loader import PipelineConfigLoader

# Lade Konfiguration für dieses Modul
config_loader = PipelineConfigLoader("02_genwiki")
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

# Zeige geladene Konfiguration
print("=" * 60)
print("📋 KONFIGURATION GELADEN (02_genwiki)")
print("=" * 60)
config_loader.print_config_summary()
print("=" * 60)

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

def extract_sections_from_md(file_path):
    """Extrahiert Überschriften und zugehörigen Inhalt aus einer Markdown-Datei."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    sections = []
    pattern = r'^(#{1,6})\s+(.+)$'
    lines = content.split('\n')
    
    current_section = None
    current_content = []
    
    for line in lines:
        heading_match = re.match(pattern, line)
        
        if heading_match:
            if current_section:
                sections.append({
                    'level': len(current_section['level']),
                    'title': current_section['title'],
                    'content': '\n'.join(current_content).strip()
                })
            
            current_section = {
                'level': heading_match.group(1),
                'title': heading_match.group(2).strip()
            }
            current_content = []
        else:
            if current_section:
                current_content.append(line)
    
    if current_section:
        sections.append({
            'level': len(current_section['level']),
            'title': current_section['title'],
            'content': '\n'.join(current_content).strip()
        })
    
    return sections

def is_content_relevant(title, content):
    """Prüft, ob der Inhalt tatsächlich Wissen vermittelt."""
    if not content.strip():
        return False
    
    prompt = f"""
Entscheide, ob dieser Abschnitt ECHTEN WISSENSINHALT enthält:

Titel: "{title}"
Inhalt: {content[:500]}...

IRRELEVANT sind:
- Inhaltsverzeichnisse, Impressum, Titel, Vorwort
- "Siehe Kapitel X", reine Verweise ohne Erklärung
- Leere oder administrative Inhalte
- Danksagungen, Literaturverzeichnisse
- Reine Listen ohne Erklärungen
- Einleitungen ohne Fachinhalt

RELEVANT sind:
- Definitionen, Erklärungen, Konzepte
- Fakten, Prozesse, Methoden
- Beispiele mit Lerninhalt
- Theorien, Modelle, Frameworks
- Praktische Anleitungen

Antworte nur: JA oder NEIN
"""
    
    response = submit_to_api(prompt)
    return response and "JA" in response.upper()

def extract_definition_term(title, content):
    """Extrahiert den optimalen Definitionsbegriff."""
    prompt = f"""
Analysiere diese Überschrift und bestimme den PRÄZISESTEN Definitionsbegriff:

Überschrift: "{title}"
Inhalt: {content[:300]}...

REGELN:
1. Entferne Nummerierungen (12.1.2.1, etc.) und Sonderzeichen (:, -, etc.)
2. Behalte spezifische Aspekte bei (z.B. "Aufgaben", "Ziele", "Methoden", "Arten")
3. Verwende den VOLLSTÄNDIGEN Begriff, nicht nur das Hauptwort
4. Bei "Aufgaben des X" → "Aufgaben des X" (nicht nur "X")
5. Bei "Berechnung von Y" → "Berechnung von Y" (nicht nur "Y")
6. Bei "Arten der Z" → "Arten der Z" (nicht nur "Z")
7. Bei reinen Begriffen ohne Zusatz (z.B. "Liquidität") → "Liquidität"

BEISPIELE:
- "12.1.2.1 Aufgaben des Cash-Management:" → "Aufgaben des Cash-Management"
- "5.2 Methoden der Risikoanalyse" → "Methoden der Risikoanalyse"
- "3.1.4 Definition Liquidität" → "Liquidität"
- "7.3 Arten von Finanzinstrumenten" → "Arten von Finanzinstrumenten"
- "4.2.1 Berechnung des Cashflows" → "Berechnung des Cashflows"
- "Netting" → "Netting"

Antworte NUR mit dem optimalen Begriff (ohne Anführungszeichen):
"""
    
    response = submit_to_api(prompt)
    if response:
        cleaned = response.strip().strip('"').strip("'")
        return cleaned if cleaned else title
    return title

def create_lexikon_prompt(optimal_title, content):
    """Erstellt einen verbesserten Prompt für die Lexikon-Erstellung."""
    prompt = f"""
Erstelle einen präzisen Wikipedia-ähnlichen Lexikon-Eintrag für: "{optimal_title}"

Originalinhalt:
{content}

ANFORDERUNGEN:
1. Der Eintrag erklärt spezifisch "{optimal_title}" (nicht das übergeordnete Thema)
2. Beginne mit einer klaren, prägnanten Definition
3. Erwähne praktische Aspekte und Anwendung
4. Halte den Fokus auf den spezifischen Begriff
5. Nutze fachlich korrekte Terminologie
6. Stil: Neutral, enzyklopädisch, sachlich
7. Sprache: Deutsch

STRUKTUR:
- Klare Definition/Einleitung von "{optimal_title}"
- Hauptmerkmale/Charakteristika
- Praktische Bedeutung/Anwendung
- Relevante Details aus dem Original-Inhalt

WICHTIG: Antworte NUR mit dem Lexikon-Eintrag, ohne zusätzliche Erklärungen oder Formatierung.
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
            {"role": "system", "content": "Du bist ein Experte für das Erstellen von Lexikon-Einträgen. Antworte immer nur mit dem gewünschten Lexikon-Eintrag, ohne zusätzliche Erklärungen."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.3
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
            print(f"Fehler bei der OpenAI-API-Anfrage (Versuch {attempt + 1}): {e}")
    
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
            {"role": "system", "content": "Du bist ein Experte für das Erstellen von Lexikon-Einträgen. Antworte immer nur mit dem gewünschten Lexikon-Eintrag, ohne zusätzliche Erklärungen."},
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
            print(f"Fehler bei der Ollama-API-Anfrage (Versuch {attempt + 1}): {e}")
    
    return None

def process_section_to_lexikon(section, max_retries=3):
    """Verbesserte Verarbeitung mit intelligenter Begriffserkennung."""
    original_title = section['title']
    content = section['content']
    
    print(f"🔍 Prüfe Relevanz von '{original_title}'...")
    if not is_content_relevant(original_title, content):
        print(f"⚠️ Sektion '{original_title}' ist nicht relevant, wird übersprungen.")
        return None
    
    print(f"🔍 Extrahiere Definitionsbegriff für '{original_title}'...")
    optimal_title = extract_definition_term(original_title, content)
    print(f"✅ Definitionsbegriff: '{optimal_title}'")
    
    print(f"📖 Erstelle Lexikon-Eintrag für: {optimal_title}")
    prompt = create_lexikon_prompt(optimal_title, content)
    
    for attempt in range(max_retries):
        response = submit_to_api(prompt)
        
        if response:
            print(f"✅ Sektion '{original_title}' → Lexikon-Eintrag '{optimal_title}'")
            return {
                'title': optimal_title,
                'original_title': original_title,
                'level': section['level'],
                'original_content': content,
                'lexikon_entry': response,
                'source': f"Section: {original_title}"
            }
        
        print(f"❌ Fehler bei Sektion '{optimal_title}', Versuch {attempt + 1}/{max_retries}")
    
    print(f"⚠️ Sektion '{original_title}' konnte nicht verarbeitet werden.")
    return None

def process_markdown_to_lexikon(md_file_path, output_json):
    """Verarbeitet eine Markdown-Datei zu einem Lexikon."""
    print(f"🔄 Starte Verarbeitung von {md_file_path}...")
    
    sections = extract_sections_from_md(md_file_path)
    print(f"📋 Gefundene Sektionen: {len(sections)}")
    
    for i, section in enumerate(sections, 1):
        indent = "  " * (section['level'] - 1)
        print(f"{i:2d}. {indent}{section['title']} (Level {section['level']})")
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    lexikon_entries = []
    skipped_sections = []
    
    for idx, section in enumerate(sections, 1):
        print(f"\n🔄 Verarbeite Sektion {idx}/{len(sections)}: {section['title']}")
        
        entry = process_section_to_lexikon(section)
        
        if entry:
            lexikon_entries.append(entry)
            print(f"✅ Sektion '{section['title']}' erfolgreich verarbeitet")
        else:
            skipped_sections.append(section['title'])
            print(f"⚠️ Sektion '{section['title']}' übersprungen")
    
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    result = {
        'metadata': {
            'source_file': str(Path(md_file_path).name),
            'total_sections': len(sections),
            'processed_sections': len(lexikon_entries),
            'skipped_sections': len(skipped_sections),
            'api_used': api_name,
            'model_used': model_name,
            'skipped_section_titles': skipped_sections
        },
        'lexikon_entries': lexikon_entries
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Verarbeitung abgeschlossen!")
    print(f"📊 Statistik:")
    print(f"   - API verwendet: {api_name}")
    print(f"   - Modell verwendet: {model_name}")
    print(f"   - Gefundene Sektionen: {len(sections)}")
    print(f"   - Erfolgreich verarbeitet: {len(lexikon_entries)}")
    print(f"   - Übersprungen: {len(skipped_sections)}")
    print(f"   - Ergebnis gespeichert in: {output_json}")

def preview_sections(md_file_path, limit=5):
    """Zeigt eine Vorschau der ersten paar Sektionen an."""
    sections = extract_sections_from_md(md_file_path)
    
    print(f"📋 Vorschau der ersten {min(limit, len(sections))} Sektionen:")
    print("=" * 50)
    
    for i, section in enumerate(sections[:limit], 1):
        indent = "  " * (section['level'] - 1)
        print(f"{i}. {indent}{section['title']} (Level {section['level']})")
        
        content_preview = section['content'][:200] + "..." if len(section['content']) > 200 else section['content']
        print(f"   Inhalt: {content_preview}")
        print("-" * 30)

def find_md_files(input_dir):
    """Findet alle Markdown-Dateien im Input-Verzeichnis."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input-Verzeichnis '{input_dir}' existiert nicht!")
        return []
    
    md_files = list(input_path.glob("*.md"))
    print(f"📁 Gefundene Markdown-Dateien: {len(md_files)}")
    for file in md_files:
        print(f"   - {file.name}")
    
    return md_files

def process_all_md_files(input_dir="INPUT", output_dir="OUTPUT"):
    """Verarbeitet alle Markdown-Dateien im Input-Verzeichnis."""
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    
    print(f"🚀 Starte Verarbeitung aller Markdown-Dateien...")
    print(f"📂 Input-Verzeichnis: {input_dir}")
    print(f"📂 Output-Verzeichnis: {output_dir}")
    print(f"🔧 API-Typ: {api_name}")
    print(f"🔧 Server: {server_url}")
    
    if not check_api_connection():
        print("❌ API-Verbindung fehlgeschlagen. Verarbeitung abgebrochen.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    md_files = find_md_files(input_dir)
    
    if not md_files:
        print("❌ Keine Markdown-Dateien gefunden!")
        return
    
    print(f"\n{'='*60}")
    print(f"Verarbeite {len(md_files)} Datei(en):")
    print(f"{'='*60}")
    
    for idx, md_file in enumerate(md_files, 1):
        print(f"\n🔄 Datei {idx}/{len(md_files)}: {md_file.name}")
        print(f"{'─'*40}")
        
        output_filename = f"lexikon_{md_file.stem}.json"
        output_path = Path(output_dir) / output_filename
        
        try:
            process_markdown_to_lexikon(str(md_file), str(output_path))
            print(f"✅ Datei {md_file.name} erfolgreich verarbeitet → {output_filename}")
            
        except Exception as e:
            print(f"❌ Fehler bei Datei {md_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"🎉 Verarbeitung aller Dateien abgeschlossen!")
    print(f"📊 Ausgabe-Dateien im Verzeichnis: {output_dir}")
    print(f"🔧 Verwendet: {api_name}")

if __name__ == "__main__":
    INPUT_DIR = "INPUT"
    OUTPUT_DIR = "OUTPUT"
    
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🔧 KONFIGURATION (aus zentraler Config):")
    print(f"   - API-Typ: {api_name}")
    print(f"   - Server: {server_url}")
    print(f"   - Modell: {model_name}")
    
    process_all_md_files(INPUT_DIR, OUTPUT_DIR)