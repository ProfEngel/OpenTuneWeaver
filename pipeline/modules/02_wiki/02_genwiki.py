import json
import requests
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# ========================================
# LOAD CENTRAL CONFIGURATION
# ========================================
sys.path.append(str(Path(__file__).parent.parent.parent))  # To main directory
from config_loader import PipelineConfigLoader

# Load configuration for this module
config_loader = PipelineConfigLoader()
module_config = config_loader.get_module_config("02_genwiki")

# Extract configuration values
API_BASE_URL = module_config.get("api_base_url", "")
API_KEY = module_config.get("api_key", "")
MODEL_NAME = module_config.get("model_name", "")

# Show loaded configuration
print("=" * 60)
print("📋 CONFIGURATION LOADED (02_genwiki)")
print("=" * 60)
config_loader.print_config_summary()
print("=" * 60)

# ========================================
# CONTENT TYPE DEFINITIONS
# ========================================

class ContentType:
    """Enum for different content types"""
    PRODUCT = "product"           # Product catalog entries with codes
    LEXIKON = "lexikon"           # Knowledge definitions
    STRUCTURED_DATA = "structured" # Tables/lists from Excel
    TECHNICAL = "technical"        # Technical documentation
    PROCESS = "process"           # Process descriptions
    THEORY = "theory"             # Theoretical concepts
    UNKNOWN = "unknown"

# ========================================
# PATTERN DEFINITIONS
# ========================================

# Product code patterns (e.g. KTS-4000, PFS-2500, ABC-123)
PRODUCT_CODE_PATTERN = r'\b[A-Z]{2,4}-\d{2,4}\b'

# Technical specification keywords (language agnostic)
TECH_SPEC_KEYWORDS = [
    'temperatur', 'temperature', 'messbereich', 'range', 'genauigkeit', 'accuracy', 
    'betriebsspannung', 'voltage', 'schutzart', 'protection', r'ip\d{2}', 'durchfluss', 
    'flow', 'frequenz', 'frequency', 'leistung', 'power', 'spannung', 'strom', 'current',
    'datenrate', 'data rate', 'auflösung', 'resolution', 'gewicht', 'weight', 
    'abmessungen', 'dimensions', 'specifications', 'specs'
]

# Structured data indicators (multilingual)
STRUCTURED_INDICATORS = [
    'bundesland', 'bundesländer', 'state', 'region', 'lieferant', 'supplier', 
    'lieferantennummer', 'supplier number', 'preis je stück', 'price per unit',
    'mitarbeiter', 'employee', 'staff', 'tabelle', 'table', 'liste', 'list', 
    'übersicht', 'overview', 'summary'
]

# ========================================
# API CONNECTION
# ========================================

def check_api_connection():
    """Checks LLM API connection."""
    try:
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 5
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat/completions", 
            json=payload, 
            headers=headers, 
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"✅ LLM API connection successful ({API_BASE_URL})")
            print(f"✅ Model '{MODEL_NAME}' is available")
            return True
        else:
            print(f"❌ LLM API not reachable (Status: {response.status_code})")
            if response.status_code == 401:
                print("🔒 Authentication failed - check API Key")
            elif response.status_code == 404:
                print("❌ Model not found - check Model Name")
            return False
            
    except requests.RequestException as e:
        print(f"❌ LLM API connection failed: {e}")
        print(f"💡 Check: Is server running at {API_BASE_URL}?")
        return False

# ========================================
# CONTENT TYPE DETECTION
# ========================================

def detect_content_type(title: str, content: str) -> Tuple[ContentType, Optional[str]]:
    """
    Detects content type and extracts product code if available.
    Returns: (ContentType, product_code or None)
    """
    title_lower = title.lower()
    content_lower = content.lower()
    
    # 1. Check for product code (highest priority)
    product_match = re.search(PRODUCT_CODE_PATTERN, title)
    if product_match:
        return ContentType.PRODUCT, product_match.group(0)
    
    # Also check content for product codes
    if re.search(PRODUCT_CODE_PATTERN, content[:200]):  # Only check beginning
        product_match = re.search(PRODUCT_CODE_PATTERN, content[:200])
        return ContentType.PRODUCT, product_match.group(0) if product_match else None
    
    # 2. Check for structured data (from Excel conversion)
    if any(indicator in title_lower for indicator in STRUCTURED_INDICATORS):
        return ContentType.STRUCTURED_DATA, None
    
    # Check for tabular structure in content
    if content.count('|') > 5 or content.count('\t') > 5:
        return ContentType.STRUCTURED_DATA, None
    
    # 3. Check for technical specifications
    tech_keyword_count = sum(1 for keyword in TECH_SPEC_KEYWORDS if keyword in content_lower)
    if tech_keyword_count >= 3:
        return ContentType.TECHNICAL, None
    
    # 4. Check for process descriptions (multilingual)
    process_keywords = ['prozess', 'process', 'ablauf', 'procedure', 'verfahren', 'method', 'methode']
    if any(keyword in title_lower for keyword in process_keywords):
        return ContentType.PROCESS, None
    
    # 5. Check for theoretical content (multilingual)
    theory_keywords = ['definition', 'theorie', 'theory', 'grundlagen', 'fundamentals', 'konzept', 'concept']
    if any(keyword in content_lower for keyword in theory_keywords):
        return ContentType.THEORY, None
    
    # 6. Standard lexicon for knowledge content (multilingual)
    lexicon_keywords = ['aufgaben', 'tasks', 'ziele', 'goals', 'arten', 'types', 'bedeutung', 'meaning', 'funktion', 'function']
    if any(keyword in title_lower for keyword in lexicon_keywords):
        return ContentType.LEXIKON, None
    
    # Default
    return ContentType.UNKNOWN, None

def is_content_relevant(title: str, content: str) -> bool:
    """Checks if content actually conveys knowledge.
    User preferred to not skip anything automatically."""
    if not content.strip():
        return False
    
    return True

# ========================================
# MARKDOWN EXTRACTION
# ========================================

def extract_sections_from_md(file_path: str) -> List[Dict[str, Any]]:
    """Extracts headings and associated content from a Markdown file."""
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
# API COMMUNICATION
# ========================================

def submit_to_api(prompt: str, retries: int = 3) -> Optional[str]:
    """Sends a request to the LLM API - LANGUAGE AGNOSTIC."""
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert at creating precise lexicon and product entries. Always respond in the same language as the source document. If the source is in German, respond in German. If the source is in English, respond in English. Always respond only with the requested entry, without additional explanations."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.3
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat/completions", 
                json=payload, 
                headers=headers, 
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return content
            else:
                print(f"❌ LLM API error {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            print(f"Error in LLM API request (attempt {attempt + 1}): {e}")
    
    return None

# ========================================
# SPECIALIZED PROCESSING FUNCTIONS
# ========================================

def process_product_entry(section: Dict[str, Any], product_code: str) -> Optional[Dict[str, Any]]:
    """Processes a product catalog entry with product code - LANGUAGE AGNOSTIC."""
    title = section['title']
    content = section['content']
    
    # Extract product name (after the code)
    product_name_match = re.search(f'{re.escape(product_code)}\\s*[-–]?\\s*(.+?)(?:\\n|$)', title + '\n' + content)
    product_name = product_name_match.group(1).strip() if product_name_match else title
    
    # Parse description and specifications (multilingual patterns)
    desc_patterns = [r'Beschreibung:\s*(.+?)(?=Technische|$)', r'Description:\s*(.+?)(?=Technical|$)', r'Description:\s*(.+?)(?=Specifications|$)']
    spec_patterns = [r'Technische\s+Spezifikationen:\s*(.+)', r'Technical\s+Specifications:\s*(.+)', r'Specifications:\s*(.+)']
    
    description = ""
    specifications = ""
    
    for pattern in desc_patterns:
        desc_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if desc_match:
            description = desc_match.group(1).strip()
            break
    
    for pattern in spec_patterns:
        spec_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if spec_match:
            specifications = spec_match.group(1).strip()
            break

    # LANGUAGE AGNOSTIC PROMPT
    prompt = f"""
Create a detailed product catalog entry for the following industrial product. 

IMPORTANT: Respond in the same language as the source document. If the source text is in German, respond in German. If the source text is in English, respond in English.

PRODUCT CODE: {product_code}
PRODUCT NAME: {product_name}

Description from original data:
{description}

Technical specifications:
{specifications}

IMPORTANT REQUIREMENTS:
1. KEEP THE EXACT PRODUCT CODE {product_code}
2. KEEP THE PRODUCT NAME "{product_name}"
3. This is a CONCRETE PRODUCT, not a general definition
4. Explain technical properties in the context of this specific product
5. Describe concrete use cases for exactly this product
6. Use the same language as the source document
7. COMPLETELY PRESERVE all facts, numbers, technical details, and the full length of the source content. DO NOT summarize or shorten the information.
8. NEVER omit any provided specification parameter.

ENTRY FORMAT (adapt to source language):
**{product_code} - {product_name}**

[Detailed product description - what this specific product does. Write a fully fleshed out text containing all details from the source.]

**Technical Features:**
[Comprehensive list of technical properties with explanations of their meaning. Do not leave out any details.]

**Application Areas:**
[Concrete application areas for this specific product]

**Special Properties:**
[What makes this product special]

IMPORTANT: This is a product catalog entry for the concrete product {product_code}, NOT a general definition! Must be highly detailed!
"""
    
    response = submit_to_api(prompt)
    
    if response:
        print(f"✅ Product entry created: {product_code} - {product_name}")
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
    """Processes structured data (from Excel conversion) - LANGUAGE AGNOSTIC."""
    title = section['title']
    content = section['content']
    
    # Clean title from numbering
    clean_title = re.sub(r'^[\d\.\s\-:]+', '', title).strip()
    
    # LANGUAGE AGNOSTIC PROMPT
    prompt = f"""
Analyze and describe this structured data from a table/list. Respond in the same language as the source document.

TITLE: {clean_title}

DATA CONTENT:
{content}

TASK:
1. Recognize the type of data (table, list, structure)
2. Describe the data structure and its meaning in extreme detail
3. Explain the practical use of this information
4. Summarize important insights while PRESERVING ALL numbers, facts, and actual data points.
5. DO NOT shorten the data. Ensure all structural and factual information is represented.

FORMAT (adapt headings to source language):
**{clean_title}**

**Data Type and Structure:**
[Comprehensive description of what kind of data this is]

**Content Meaning:**
[Detailed explanation of what this data means in context, citing all important metrics/numbers]

**Practical Use:**
[What this data is used for]

**Important Insights:**
[Detailed summary of key points and ALL relevant data provided]
"""
    
    response = submit_to_api(prompt)
    
    if response:
        print(f"✅ Structured data entry created: {clean_title}")
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
    """Processes technical documentation - LANGUAGE AGNOSTIC."""
    title = section['title']
    content = section['content']
    
    # Clean title
    clean_title = re.sub(r'^[\d\.\s\-:]+', '', title).strip()
    
    # LANGUAGE AGNOSTIC PROMPT
    prompt = f"""
Create a technical documentation entry. Respond in the same language as the source document.

TOPIC: {clean_title}

TECHNICAL CONTENT:
{content}

REQUIREMENTS:
1. Explain technical concepts precisely and understandably, but DO NOT shorten the text.
2. Keep all technical terminology and explain it.
3. Structure by: Definition, Functionality, Application.
4. Mention ALL concrete technical parameters, numbers, and specifications provided. Do not summarize them away.

FORMAT (adapt headings to source language):
**{clean_title}**

**Technical Definition:**
[Detailed and precise technical description preserving full source length]

**Functionality:**
[Comprehensive explanation of how it works technically]

**Technical Parameters:**
[Exhaustive inclusion of ALL specifications and values found in the text]

**Practical Application:**
[Where and how it is used based on all provided details]
"""
    
    response = submit_to_api(prompt)
    
    if response:
        print(f"✅ Technical entry created: {clean_title}")
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
    """Processes process descriptions - LANGUAGE AGNOSTIC."""
    title = section['title']
    content = section['content']
    
    # Clean title
    clean_title = re.sub(r'^[\d\.\s\-:]+', '', title).strip()
    
    # LANGUAGE AGNOSTIC PROMPT
    prompt = f"""
Create a structured process description. Respond in the same language as the source document.

PROCESS: {clean_title}

PROCESS CONTENT:
{content}

REQUIREMENTS:
1. COMPLETELY PRESERVE all steps, preconditions, details, and warnings.
2. DO NOT summarize or shorten the process. It must remain fully actionable and comprehensive.

FORMAT (adapt headings to source language):
**{clean_title}**

**Process Overview:**
[Detailed summary of the process context]

**Process Steps:**
[Exhaustive and structured presentation of ALL steps without skipping any context]

**Prerequisites:**
[Everything that is needed]

**Result:**
[Precise goal/output]

**Important Notes:**
[All specifics, best practices, and warnings from the text]
"""
    
    response = submit_to_api(prompt)
    
    if response:
        print(f"✅ Process entry created: {clean_title}")
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
    """Processes theoretical concepts - LANGUAGE AGNOSTIC."""
    title = section['title']
    content = section['content']
    
    # Extract optimal term
    clean_title = extract_definition_term(title, content)
    
    # LANGUAGE AGNOSTIC PROMPT
    prompt = f"""
Create a scientific lexicon entry. Respond in the same language as the source document.

CONCEPT: {clean_title}

ORIGINAL CONTENT:
{content}

REQUIREMENTS:
1. Start with a precise, highly detailed scientific definition.
2. Explain theoretical foundations extensively.
3. Mention practical applications.
4. Use academic, neutral language.
5. DO NOT shorten or summarize the original content. Ensure ALL theoretical arguments, nuances, facts, and context from the original text are fully incorporated in your answer.

FORMAT (adapt headings to source language):
**{clean_title}**

[Comprehensive scientific definition]

**Theoretical Foundations:**
[Exhaustive explanation of theory preserving all original length and arguments]

**Practical Significance:**
[Application in practice detailing all original points]

**Related Concepts:**
[If relevant, explain all mentioned related concepts]
"""
    
    response = submit_to_api(prompt)
    
    if response:
        print(f"✅ Theory entry created: {clean_title}")
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
    """Processes standard lexicon entries (previous functionality) - LANGUAGE AGNOSTIC."""
    title = section['title']
    content = section['content']
    
    # Extract optimal term
    optimal_title = extract_definition_term(title, content)
    
    prompt = create_lexikon_prompt(optimal_title, content)
    response = submit_to_api(prompt)
    
    if response:
        print(f"✅ Lexicon entry created: {optimal_title}")
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
    """Extracts the optimal definition term - LANGUAGE AGNOSTIC."""
    # LANGUAGE AGNOSTIC PROMPT
    prompt = f"""
Analyze this heading and determine the MOST PRECISE definition term. Respond in the same language as the source.

Heading: "{title}"
Content: {content[:300]}...

RULES:
1. Remove numbering (12.1.2.1, etc.) and special characters (:, -, etc.)
2. Keep specific aspects (e.g. "Tasks", "Goals", "Methods", "Types", "Aufgaben", "Ziele", "Methoden", "Arten")
3. Use the COMPLETE term, not just the main word
4. For "Tasks of X" / "Aufgaben des X" → "Tasks of X" / "Aufgaben des X" (not just "X")
5. For "Calculation of Y" / "Berechnung von Y" → "Calculation of Y" / "Berechnung von Y" (not just "Y")
6. For "Types of Z" / "Arten der Z" → "Types of Z" / "Arten der Z" (not just "Z")
7. For pure terms without addition (e.g. "Liquidity" / "Liquidität") → "Liquidity" / "Liquidität"

EXAMPLES:
- "12.1.2.1 Tasks of Cash Management:" → "Tasks of Cash Management" / "Aufgaben des Cash-Management"
- "5.2 Methods of Risk Analysis" → "Methods of Risk Analysis" / "Methoden der Risikoanalyse"
- "3.1.4 Definition Liquidity" → "Liquidity" / "Liquidität"
- "7.3 Types of Financial Instruments" → "Types of Financial Instruments" / "Arten von Finanzinstrumenten"
- "4.2.1 Calculation of Cashflow" → "Calculation of Cashflow" / "Berechnung des Cashflows"
- "Netting" → "Netting"

Respond ONLY with the optimal term (without quotes):
"""
    
    response = submit_to_api(prompt)
    if response:
        cleaned = response.strip().strip('"').strip("'")
        return cleaned if cleaned else title
    return title

def create_lexikon_prompt(optimal_title: str, content: str) -> str:
    """Creates a prompt for standard lexicon entries - LANGUAGE AGNOSTIC."""
    prompt = f"""
Create a precise Wikipedia-like lexicon entry. Respond in the same language as the source document.

Term: "{optimal_title}"

Original content:
{content}

REQUIREMENTS:
1. The entry explains specifically "{optimal_title}" (not the broader topic)
2. Start with a clear, comprehensive definition
3. Mention practical aspects and application
4. Keep focus on the specific term
5. Use professionally correct terminology
6. Style: Neutral, informative, factual
7. Language: Same as the source document
8. EXTREMELY IMPORTANT: DO NOT SUMMARIZE OR SHORTEN. You must preserve ALL details, facts, numbers, and the FULL semantic length of the original content. Your output must be exhaustive.

STRUCTURE:
- Comprehensive definition/introduction of "{optimal_title}"
- Exhaustive main features/characteristics
- Detailed practical significance/application
- ALL relevant details from the original content, leaving nothing out.

IMPORTANT: Respond ONLY with the highly detailed lexicon entry, without additional explanations or formatting.
"""
    return prompt

# ========================================
# MAIN PROCESSING FUNCTION
# ========================================

def process_section_to_entry(section: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Intelligent processing of a section based on content type."""
    title = section['title']
    content = section['content']
    
    # Check relevance
    if not is_content_relevant(title, content):
        print(f"⚠️ Section '{title}' is not relevant, skipping.")
        return None
    
    # Detect content type
    content_type, product_code = detect_content_type(title, content)
    print(f"🔍 Detected type for '{title}': {content_type}" + (f" (Code: {product_code})" if product_code else ""))
    
    # Process based on type
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
            # Fallback to standard lexicon
            entry = process_lexikon_entry(section)
    
    except Exception as e:
        print(f"❌ Error processing '{title}': {e}")
    
    return entry

# ========================================
# FILE PROCESSING
# ========================================

def process_markdown_to_lexikon(md_file_path: str, output_json: str):
    """Processes a Markdown file to a lexicon."""
    print(f"🔄 Starting processing of {md_file_path}...")
    
    sections = extract_sections_from_md(md_file_path)
    print(f"📋 Found sections: {len(sections)}")
    
    # Show overview
    for i, section in enumerate(sections, 1):
        indent = "  " * (section['level'] - 1)
        # Detect type for display
        content_type, product_code = detect_content_type(section['title'], section['content'])
        type_indicator = f" [{content_type}]"
        if product_code:
            type_indicator += f" ({product_code})"
        print(f"{i:2d}. {indent}{section['title']}{type_indicator}")
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    entries = []
    skipped_sections = []
    
    # Statistics by type
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
        print(f"\n🔄 Processing section {idx}/{len(sections)}: {section['title']}")
        
        entry = process_section_to_entry(section)
        
        if entry:
            entries.append(entry)
            entry_type = entry.get('type', 'unknown')
            type_stats[entry_type] = type_stats.get(entry_type, 0) + 1
            print(f"✅ Successfully processed as: {entry_type}")
        else:
            skipped_sections.append(section['title'])
            print(f"⚠️ Section skipped")
    
    # Create metadata
    result = {
        'metadata': {
            'source_file': str(Path(md_file_path).name),
            'processed_at': datetime.now().isoformat(),
            'total_sections': len(sections),
            'processed_sections': len(entries),
            'skipped_sections': len(skipped_sections),
            'api_used': "LLM API",
            'model_used': MODEL_NAME,
            'entry_statistics': type_stats,
            'skipped_section_titles': skipped_sections,
            'language_handling': 'agnostic'
        },
        'lexikon_entries': entries
    }
    
    # Save result
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Processing completed!")
    print(f"📊 Statistics:")
    print(f"   - API used: LLM API")
    print(f"   - Model used: {MODEL_NAME}")
    print(f"   - Found sections: {len(sections)}")
    print(f"   - Successfully processed: {len(entries)}")
    print(f"   - Entry types:")
    for entry_type, count in type_stats.items():
        if count > 0:
            print(f"     • {entry_type}: {count}")
    print(f"   - Skipped: {len(skipped_sections)}")
    print(f"   - Result saved to: {output_json}")
    print(f"   - Language handling: Agnostic (preserves document language)")

def preview_sections(md_file_path: str, limit: int = 5):
    """Shows a preview of the first sections with type detection."""
    sections = extract_sections_from_md(md_file_path)
    
    print(f"📋 Preview of first {min(limit, len(sections))} sections:")
    print("=" * 50)
    
    for i, section in enumerate(sections[:limit], 1):
        indent = "  " * (section['level'] - 1)
        content_type, product_code = detect_content_type(section['title'], section['content'])
        
        print(f"{i}. {indent}{section['title']} (Level {section['level']})")
        print(f"   Type: {content_type}" + (f" [Code: {product_code}]" if product_code else ""))
        
        content_preview = section['content'][:200] + "..." if len(section['content']) > 200 else section['content']
        print(f"   Content: {content_preview}")
        print("-" * 30)

def find_md_files(input_dir: str) -> List[Path]:
    """Finds all Markdown files in the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory '{input_dir}' does not exist!")
        return []
    
    md_files = list(input_path.glob("*.md"))
    print(f"📁 Found Markdown files: {len(md_files)}")
    for file in md_files:
        print(f"   - {file.name}")
    
    return md_files

def process_all_md_files(input_dir: str = "INPUT", output_dir: str = "OUTPUT"):
    """Processes all Markdown files in the input directory."""
    print(f"🚀 Starting processing of all Markdown files...")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"🔧 API type: LLM API")
    print(f"🔧 Server: {API_BASE_URL}")
    print(f"🌐 Language handling: Agnostic (preserves document language)")
    
    if not check_api_connection():
        print("❌ API connection failed. Processing aborted.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    md_files = find_md_files(input_dir)
    
    if not md_files:
        print("❌ No Markdown files found!")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {len(md_files)} file(s):")
    print(f"{'='*60}")
    
    # Total statistics
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
        print(f"\n🔄 File {idx}/{len(md_files)}: {md_file.name}")
        print(f"{'─'*40}")
        
        output_filename = f"lexikon_{md_file.stem}.json"
        output_path = Path(output_dir) / output_filename
        
        try:
            process_markdown_to_lexikon(str(md_file), str(output_path))
            
            # Load statistics for overall view
            with open(output_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
                stats = result.get('metadata', {}).get('entry_statistics', {})
                for key, value in stats.items():
                    total_stats[key] = total_stats.get(key, 0) + value
            
            print(f"✅ File {md_file.name} successfully processed → {output_filename}")
            
        except Exception as e:
            print(f"❌ Error with file {md_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"🎉 Processing of all files completed!")
    print(f"📊 Total statistics of all entries:")
    for entry_type, count in total_stats.items():
        if count > 0:
            print(f"   • {entry_type}: {count}")
    print(f"📂 Output files in directory: {output_dir}")
    print(f"🔧 Used: LLM API")
    print(f"🌐 Language preservation: ✅ Document language preserved")

# ========================================
# MAIN PROGRAM
# ========================================

if __name__ == "__main__":
    INPUT_DIR = "INPUT"
    OUTPUT_DIR = "OUTPUT"
    
    print(f"🔧 CONFIGURATION (from central config):")
    print(f"   - API type: LLM API")
    print(f"   - Server: {API_BASE_URL}")
    print(f"   - Model: {MODEL_NAME}")
    print(f"   - Language handling: Agnostic (preserves document language)")
    
    # Optional: Preview for a specific file
    # preview_sections("INPUT/Product Catalog Relmiad AG.md", limit=10)
    
    # Process all files
    process_all_md_files(INPUT_DIR, OUTPUT_DIR)