import json
import requests
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

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

# ========================================
# CONTENT-TYP DEFINITIONEN
# ========================================

class ContentType:
    """Enum für verschiedene Content-Typen"""
    PRODUCT = "product"           # Produktkatalog-Einträge mit Codes
    LEXIKON = "lexikon"           # Wissensdefinitionen
    STRUCTURED_DATA = "structured" # Tabellen/Listen aus Excel
    TECHNICAL = "technical"        # Technische Dokumentation
    PROCESS = "process"           # Prozessbeschreibungen
    THEORY = "theory"             # Theoretische Konzepte
    UNKNOWN = "unknown"

# ========================================
# PATTERN DEFINITIONEN
# ========================================

# Produktcode-Patterns (z.B. KTS-4000, PFS-2500, ABC-123)
PRODUCT_CODE_PATTERN = r'\b[A-Z]{2,4}-\d{2,4}\b'

# Technische Spezifikations-Keywords
TECH_SPEC_KEYWORDS = [
    'temperaturbereich', 'messbereich', 'genauigkeit', 'betriebsspannung',
    'schutzart', 'ip\d{2}', 'durchfluss', 'frequenz', 'leistung',
    'spannung', 'strom', 'datenrate', 'auflösung', 'gewicht', 'abmessungen'
]

# Strukturdaten-Indikatoren
STRUCTURED_INDICATORS = [
    'bundesland', 'bundesländer', 'lieferant', 'lieferantennummer',
    'preis je stück', 'mitarbeiter', 'tabelle', 'liste', 'übersicht'
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
# CONTENT-TYP ERKENNUNG
# ========================================

def detect_content_type(title: str, content: str) -> Tuple[ContentType, Optional[str]]:
    """
    Erkennt den Content-Typ und extrahiert ggf. Produktcode.
    Returns: (ContentType, product_code or None)
    """
    title_lower = title.lower()
    content_lower = content.lower()
    
    # 1. Prüfe auf Produktcode (höchste Priorität)
    product_match = re.search(PRODUCT_CODE_PATTERN, title)
    if product_match:
        return ContentType.PRODUCT, product_match.group(0)
    
    # Prüfe auch im Content nach Produktcodes
    if re.search(PRODUCT_CODE_PATTERN, content[:200]):  # Nur Anfang prüfen
        product_match = re.search(PRODUCT_CODE_PATTERN, content[:200])
        return ContentType.PRODUCT, product_match.group(0) if product_match else None
    
    # 2. Prüfe auf strukturierte Daten (aus Excel-Konvertierung)
    if any(indicator in title_lower for indicator in STRUCTURED_INDICATORS):
        return ContentType.STRUCTURED_DATA, None
    
    # Prüfe auf tabellarische Struktur im Content
    if content.count('|') > 5 or content.count('\t') > 5:
        return ContentType.STRUCTURED_DATA, None
    
    # 3. Prüfe auf technische Spezifikationen
    tech_keyword_count = sum(1 for keyword in TECH_SPEC_KEYWORDS if keyword in content_lower)
    if tech_keyword_count >= 3:
        return ContentType.TECHNICAL, None
    
    # 4. Prüfe auf Prozessbeschreibungen
    if any(keyword in title_lower for keyword in ['prozess', 'ablauf', 'verfahren', 'methode']):
        return ContentType.PROCESS, None
    
    # 5. Prüfe auf theoretische Inhalte
    if any(keyword in content_lower for keyword in ['definition', 'theorie', 'grundlagen', 'konzept']):
        return ContentType.THEORY, None
    
    # 6. Standard-Lexikon für Wissensinhalte
    if any(keyword in title_lower for keyword in ['aufgaben', 'ziele', 'arten', 'bedeutung', 'funktion']):
        return ContentType.LEXIKON, None
    
    # Default
    return ContentType.UNKNOWN, None

def is_content_relevant(title: str, content: str) -> bool:
    """Prüft, ob der Inhalt tatsächlich Wissen vermittelt."""
    if not content.strip():
        return False
    
    # Irrelevante Titel-Keywords
    irrelevant_keywords = [
        'inhaltsverzeichnis', 'impressum', 'literaturverzeichnis',
        'anhang', 'index', 'glossar', 'danksagung', 'vorwort'
    ]
    
    title_lower = title.lower()
    if any(keyword in title_lower for keyword in irrelevant_keywords):
        return False
    
    # Prüfe auf Mindestinhalt
    word_count = len(content.split())
    if word_count < 10:
        return False
    
    # Prüfe auf reinen Verweis-Content
    if 'siehe kapitel' in content.lower() and word_count < 30:
        return False
    
    return True

# ========================================
# MARKDOWN EXTRAKTION
# ========================================

def extract_sections_from_md(file_path: str) -> List[Dict[str, Any]]:
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

# ========================================
# API KOMMUNIKATION
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
            {"role": "system", "content": "Du bist ein Experte für das Erstellen von präzisen Lexikon- und Produkteinträgen. Antworte immer nur mit dem gewünschten Eintrag, ohne zusätzliche Erklärungen."},
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

def submit_to_ollama_api(prompt: str, retries: int = 3) -> Optional[str]:
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
            {"role": "system", "content": "Du bist ein Experte für das Erstellen von präzisen Lexikon- und Produkteinträgen. Antworte immer nur mit dem gewünschten Eintrag, ohne zusätzliche Erklärungen."},
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

# ========================================
# SPEZIALISIERTE VERARBEITUNGSFUNKTIONEN
# ========================================

def process_product_entry(section: Dict[str, Any], product_code: str) -> Optional[Dict[str, Any]]:
    """Verarbeitet einen Produktkatalog-Eintrag mit Produktcode."""
    title = section['title']
    content = section['content']
    
    # Extrahiere Produktnamen (nach dem Code)
    product_name_match = re.search(f'{re.escape(product_code)}\\s*[-–]?\\s*(.+?)(?:\\n|$)', title + '\n' + content)
    product_name = product_name_match.group(1).strip() if product_name_match else title
    
    # Parse Beschreibung und Spezifikationen
    desc_match = re.search(r'Beschreibung:\s*(.+?)(?=Technische|$)', content, re.DOTALL | re.IGNORECASE)
    spec_match = re.search(r'Technische\s+Spezifikationen:\s*(.+)', content, re.DOTALL | re.IGNORECASE)
    
    description = desc_match.group(1).strip() if desc_match else ""
    specifications = spec_match.group(1).strip() if spec_match else ""
    
    prompt = f"""
Erstelle einen detaillierten Produktkatalog-Eintrag für folgendes Industrieprodukt:

PRODUKTCODE: {product_code}
PRODUKTNAME: {product_name}

Beschreibung aus Originaldaten:
{description}

Technische Spezifikationen:
{specifications}

WICHTIGE ANFORDERUNGEN:
1. BEHALTE DEN EXAKTEN PRODUKTCODE {product_code} BEI
2. BEHALTE DEN PRODUKTNAMEN "{product_name}" BEI
3. Dies ist ein KONKRETES PRODUKT, keine allgemeine Definition
4. Erkläre die technischen Eigenschaften im Kontext dieses spezifischen Produkts
5. Beschreibe konkrete Anwendungsfälle für genau dieses Produkt

FORMAT DES EINTRAGS:
**{product_code} - {product_name}**

[Detaillierte Produktbeschreibung - was macht dieses spezifische Produkt]

**Technische Merkmale:**
[Liste der technischen Eigenschaften mit Erklärungen ihrer Bedeutung]

**Anwendungsbereiche:**
[Konkrete Einsatzgebiete für dieses spezifische Produkt]

**Besondere Eigenschaften:**
[Was zeichnet dieses Produkt speziell aus]

WICHTIG: Dies ist ein Produktkatalog-Eintrag für das konkrete Produkt {product_code}, NICHT eine allgemeine Definition!
"""
    
    response = submit_to_api(prompt)
    
    if response:
        print(f"✅ Produkteintrag erstellt: {product_code} - {product_name}")
        return {
            'title': f"{product_code} - {product_name}",
            'product_code': product_code,
            'type': 'product',
            'level': section['level'],
            'original_title': section['title'],
            'original_content': content,
            'lexikon_entry': response,
            'source': f"Product: {title}"
        }
    
    return None

def process_structured_data_entry(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Verarbeitet strukturierte Daten (aus Excel-Konvertierung)."""
    title = section['title']
    content = section['content']
    
    # Bereinige Titel von Nummerierungen
    clean_title = re.sub(r'^[\d\.\s\-:]+', '', title).strip()
    
    prompt = f"""
Analysiere und beschreibe diese strukturierten Daten aus einer Tabelle/Liste:

TITEL: {clean_title}

DATENINHALT:
{content}

AUFGABE:
1. Erkenne die Art der Daten (Tabelle, Liste, Struktur)
2. Beschreibe die Datenstruktur und deren Bedeutung
3. Erkläre den praktischen Nutzen dieser Informationen
4. Fasse wichtige Erkenntnisse zusammen

FORMAT:
**{clean_title}**

**Datentyp und Struktur:**
[Beschreibung um welche Art von Daten es sich handelt]

**Inhaltliche Bedeutung:**
[Was bedeuten diese Daten im Kontext]

**Praktische Verwendung:**
[Wofür werden diese Daten genutzt]

**Wichtige Erkenntnisse:**
[Zusammenfassung der Kernpunkte]
"""
    
    response = submit_to_api(prompt)
    
    if response:
        print(f"✅ Strukturdaten-Eintrag erstellt: {clean_title}")
        return {
            'title': clean_title,
            'type': 'structured_data',
            'level': section['level'],
            'original_title': section['title'],
            'original_content': content,
            'lexikon_entry': response,
            'source': f"Structured Data: {title}"
        }
    
    return None

def process_technical_entry(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Verarbeitet technische Dokumentation."""
    title = section['title']
    content = section['content']
    
    # Bereinige Titel
    clean_title = re.sub(r'^[\d\.\s\-:]+', '', title).strip()
    
    prompt = f"""
Erstelle einen technischen Dokumentations-Eintrag für:

THEMA: {clean_title}

TECHNISCHER INHALT:
{content}

ANFORDERUNGEN:
1. Erkläre technische Konzepte präzise und verständlich
2. Behalte Fachterminologie bei, erkläre sie aber
3. Strukturiere nach: Definition, Funktionsweise, Anwendung
4. Nenne konkrete technische Parameter wenn vorhanden

FORMAT:
**{clean_title}**

**Technische Definition:**
[Präzise technische Beschreibung]

**Funktionsweise:**
[Wie funktioniert es technisch]

**Technische Parameter:**
[Wichtige Kennzahlen und Spezifikationen]

**Praktische Anwendung:**
[Wo und wie wird es eingesetzt]
"""
    
    response = submit_to_api(prompt)
    
    if response:
        print(f"✅ Technischer Eintrag erstellt: {clean_title}")
        return {
            'title': clean_title,
            'type': 'technical',
            'level': section['level'],
            'original_title': section['title'],
            'original_content': content,
            'lexikon_entry': response,
            'source': f"Technical: {title}"
        }
    
    return None

def process_process_entry(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Verarbeitet Prozessbeschreibungen."""
    title = section['title']
    content = section['content']
    
    # Bereinige Titel
    clean_title = re.sub(r'^[\d\.\s\-:]+', '', title).strip()
    
    prompt = f"""
Erstelle eine strukturierte Prozessbeschreibung für:

PROZESS: {clean_title}

PROZESSINHALT:
{content}

FORMAT:
**{clean_title}**

**Prozessübersicht:**
[Kurze Zusammenfassung des Prozesses]

**Ablaufschritte:**
[Strukturierte Darstellung der Schritte]

**Voraussetzungen:**
[Was wird benötigt]

**Ergebnis:**
[Was ist das Ziel/Output]

**Wichtige Hinweise:**
[Besonderheiten, Best Practices]
"""
    
    response = submit_to_api(prompt)
    
    if response:
        print(f"✅ Prozess-Eintrag erstellt: {clean_title}")
        return {
            'title': clean_title,
            'type': 'process',
            'level': section['level'],
            'original_title': section['title'],
            'original_content': content,
            'lexikon_entry': response,
            'source': f"Process: {title}"
        }
    
    return None

def process_theory_entry(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Verarbeitet theoretische Konzepte."""
    title = section['title']
    content = section['content']
    
    # Extrahiere optimalen Begriff
    clean_title = extract_definition_term(title, content)
    
    prompt = f"""
Erstelle einen wissenschaftlichen Lexikon-Eintrag für:

KONZEPT: {clean_title}

ORIGINALINHALT:
{content}

ANFORDERUNGEN:
1. Beginne mit einer präzisen wissenschaftlichen Definition
2. Erkläre theoretische Grundlagen
3. Nenne praktische Anwendungen
4. Verwende akademische, neutrale Sprache

FORMAT:
**{clean_title}**

[Wissenschaftliche Definition]

**Theoretische Grundlagen:**
[Erklärung der Theorie]

**Praktische Bedeutung:**
[Anwendung in der Praxis]

**Verwandte Konzepte:**
[Falls relevant]
"""
    
    response = submit_to_api(prompt)
    
    if response:
        print(f"✅ Theorie-Eintrag erstellt: {clean_title}")
        return {
            'title': clean_title,
            'type': 'theory',
            'level': section['level'],
            'original_title': section['title'],
            'original_content': content,
            'lexikon_entry': response,
            'source': f"Theory: {title}"
        }
    
    return None

def process_lexikon_entry(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Verarbeitet Standard-Lexikon-Einträge (bisherige Funktionalität)."""
    title = section['title']
    content = section['content']
    
    # Extrahiere optimalen Begriff
    optimal_title = extract_definition_term(title, content)
    
    prompt = create_lexikon_prompt(optimal_title, content)
    response = submit_to_api(prompt)
    
    if response:
        print(f"✅ Lexikon-Eintrag erstellt: {optimal_title}")
        return {
            'title': optimal_title,
            'type': 'lexikon',
            'level': section['level'],
            'original_title': section['title'],
            'original_content': content,
            'lexikon_entry': response,
            'source': f"Section: {title}"
        }
    
    return None

def extract_definition_term(title: str, content: str) -> str:
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

def create_lexikon_prompt(optimal_title: str, content: str) -> str:
    """Erstellt einen Prompt für Standard-Lexikon-Einträge."""
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

# ========================================
# HAUPTVERARBEITUNGSFUNKTION
# ========================================

def process_section_to_entry(section: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Intelligente Verarbeitung einer Sektion basierend auf Content-Typ."""
    title = section['title']
    content = section['content']
    
    # Prüfe Relevanz
    if not is_content_relevant(title, content):
        print(f"⚠️ Sektion '{title}' ist nicht relevant, wird übersprungen.")
        return None
    
    # Erkenne Content-Typ
    content_type, product_code = detect_content_type(title, content)
    print(f"🔍 Erkannter Typ für '{title}': {content_type}" + (f" (Code: {product_code})" if product_code else ""))
    
    # Verarbeite basierend auf Typ
    entry = None
    
    try:
        if content_type == ContentType.PRODUCT:
            entry = process_product_entry(section, product_code)
        elif content_type == ContentType.STRUCTURED_DATA:
            entry = process_structured_data_entry(section)
        elif content_type == ContentType.TECHNICAL:
            entry = process_technical_entry(section)
        elif content_type == ContentType.PROCESS:
            entry = process_process_entry(section)
        elif content_type == ContentType.THEORY:
            entry = process_theory_entry(section)
        elif content_type == ContentType.LEXIKON:
            entry = process_lexikon_entry(section)
        else:
            # Fallback auf Standard-Lexikon
            entry = process_lexikon_entry(section)
    
    except Exception as e:
        print(f"❌ Fehler bei Verarbeitung von '{title}': {e}")
    
    return entry

# ========================================
# DATEIVERARBEITUNG
# ========================================

def process_markdown_to_lexikon(md_file_path: str, output_json: str):
    """Verarbeitet eine Markdown-Datei zu einem Lexikon."""
    print(f"🔄 Starte Verarbeitung von {md_file_path}...")
    
    sections = extract_sections_from_md(md_file_path)
    print(f"📋 Gefundene Sektionen: {len(sections)}")
    
    # Zeige Übersicht
    for i, section in enumerate(sections, 1):
        indent = "  " * (section['level'] - 1)
        # Erkenne Typ für Anzeige
        content_type, product_code = detect_content_type(section['title'], section['content'])
        type_indicator = f" [{content_type}]"
        if product_code:
            type_indicator += f" ({product_code})"
        print(f"{i:2d}. {indent}{section['title']}{type_indicator}")
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    entries = []
    skipped_sections = []
    
    # Statistik nach Typ
    type_stats = {
        'product': 0,
        'structured_data': 0,
        'technical': 0,
        'process': 0,
        'theory': 0,
        'lexikon': 0,
        'unknown': 0
    }
    
    for idx, section in enumerate(sections, 1):
        print(f"\n🔄 Verarbeite Sektion {idx}/{len(sections)}: {section['title']}")
        
        entry = process_section_to_entry(section)
        
        if entry:
            entries.append(entry)
            entry_type = entry.get('type', 'unknown')
            type_stats[entry_type] = type_stats.get(entry_type, 0) + 1
            print(f"✅ Erfolgreich verarbeitet als: {entry_type}")
        else:
            skipped_sections.append(section['title'])
            print(f"⚠️ Sektion übersprungen")
    
    # Erstelle Metadaten
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    result = {
        'metadata': {
            'source_file': str(Path(md_file_path).name),
            'processed_at': datetime.now().isoformat(),
            'total_sections': len(sections),
            'processed_sections': len(entries),
            'skipped_sections': len(skipped_sections),
            'api_used': api_name,
            'model_used': model_name,
            'entry_statistics': type_stats,
            'skipped_section_titles': skipped_sections
        },
        'lexikon_entries': entries
    }
    
    # Speichere Ergebnis
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Verarbeitung abgeschlossen!")
    print(f"📊 Statistik:")
    print(f"   - API verwendet: {api_name}")
    print(f"   - Modell verwendet: {model_name}")
    print(f"   - Gefundene Sektionen: {len(sections)}")
    print(f"   - Erfolgreich verarbeitet: {len(entries)}")
    print(f"   - Eintragstypen:")
    for entry_type, count in type_stats.items():
        if count > 0:
            print(f"     • {entry_type}: {count}")
    print(f"   - Übersprungen: {len(skipped_sections)}")
    print(f"   - Ergebnis gespeichert in: {output_json}")

def preview_sections(md_file_path: str, limit: int = 5):
    """Zeigt eine Vorschau der ersten Sektionen mit Typ-Erkennung."""
    sections = extract_sections_from_md(md_file_path)
    
    print(f"📋 Vorschau der ersten {min(limit, len(sections))} Sektionen:")
    print("=" * 50)
    
    for i, section in enumerate(sections[:limit], 1):
        indent = "  " * (section['level'] - 1)
        content_type, product_code = detect_content_type(section['title'], section['content'])
        
        print(f"{i}. {indent}{section['title']} (Level {section['level']})")
        print(f"   Typ: {content_type}" + (f" [Code: {product_code}]" if product_code else ""))
        
        content_preview = section['content'][:200] + "..." if len(section['content']) > 200 else section['content']
        print(f"   Inhalt: {content_preview}")
        print("-" * 30)

def find_md_files(input_dir: str) -> List[Path]:
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

def process_all_md_files(input_dir: str = "INPUT", output_dir: str = "OUTPUT"):
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
    
    # Gesamtstatistik
    total_stats = {
        'product': 0,
        'structured_data': 0,
        'technical': 0,
        'process': 0,
        'theory': 0,
        'lexikon': 0,
        'unknown': 0
    }
    
    for idx, md_file in enumerate(md_files, 1):
        print(f"\n🔄 Datei {idx}/{len(md_files)}: {md_file.name}")
        print(f"{'─'*40}")
        
        output_filename = f"lexikon_{md_file.stem}.json"
        output_path = Path(output_dir) / output_filename
        
        try:
            process_markdown_to_lexikon(str(md_file), str(output_path))
            
            # Lade Statistik für Gesamtübersicht
            with open(output_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
                stats = result.get('metadata', {}).get('entry_statistics', {})
                for key, value in stats.items():
                    total_stats[key] = total_stats.get(key, 0) + value
            
            print(f"✅ Datei {md_file.name} erfolgreich verarbeitet → {output_filename}")
            
        except Exception as e:
            print(f"❌ Fehler bei Datei {md_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"🎉 Verarbeitung aller Dateien abgeschlossen!")
    print(f"📊 Gesamtstatistik aller Einträge:")
    for entry_type, count in total_stats.items():
        if count > 0:
            print(f"   • {entry_type}: {count}")
    print(f"📂 Ausgabe-Dateien im Verzeichnis: {output_dir}")
    print(f"🔧 Verwendet: {api_name}")

# ========================================
# HAUPTPROGRAMM
# ========================================

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
    
    # Optional: Vorschau für eine spezifische Datei
    # preview_sections("INPUT/Produktkatalog Relmiad AG.md", limit=10)
    
    # Verarbeite alle Dateien
    process_all_md_files(INPUT_DIR, OUTPUT_DIR)