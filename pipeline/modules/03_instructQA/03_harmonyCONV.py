import json
import requests
import random
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# ========================================
# LOAD CENTRAL CONFIGURATION
# ========================================
sys.path.append(str(Path(__file__).parent.parent.parent))  # To main directory
from config_loader import PipelineConfigLoader

# Load configuration for this module
config_loader = PipelineConfigLoader("03_instructQA")
config = config_loader.get_api_config()

# Extract configuration values
USE_OPENAI_API = config.get("use_openai_api", True)
OPENAI_BASE_URL = config.get("openai_base_url", "http://localhost:11434/v1")
OPENAI_API_KEY = config.get("openai_api_key", "ollama")
OPENAI_MODEL_NAME = config.get("openai_model_name", "gpt-oss:20b")
OLLAMA_SERVER_URL = config.get("ollama_server_url", "http://localhost:11434")
OLLAMA_API_KEY = config.get("ollama_api_key", "ollama")
OLLAMA_MODEL_NAME = config.get("ollama_model_name", "gpt-oss:20b")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_SERVER_URL}/api/chat"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_SERVER_URL}/api/tags"

# Directories
INPUT_DIR = "INPUT"
OUTPUT_DIR = "OUTPUT"
OUTPUT_FILENAME = "thinking_dataset.jsonl"

# Show loaded configuration
print("=" * 60)
print("🧠 CONFIGURATION LOADED (Thinking Dataset Format)")
print("=" * 60)
config_loader.print_config_summary()
print("=" * 60)

# ========================================
# EXTENDED QUESTION TYPES
# ========================================

QUESTION_TYPES = [
    {
        "type": "definition",
        "templates": [
            "What is {title}?",
            "Can you explain {title}?",
            "Define {title}.",
            "What is meant by {title}?",
            "Explain the concept of {title}."
        ]
    },
    {
        "type": "detailed_explanation",
        "templates": [
            "Describe {title} in detail.",
            "Give a detailed explanation of {title}.",
            "What should one know about {title}?",
            "Explain {title} to me in more detail.",
            "What are the important aspects of {title}?"
        ]
    },
    {
        "type": "characteristics",
        "templates": [
            "What characteristics does {title} have?",
            "What are the main properties of {title}?",
            "What characteristics distinguish {title}?",
            "What are the essential properties of {title}?",
            "What special features characterize {title}?"
        ]
    },
    {
        "type": "application",
        "templates": [
            "What is {title} used for?",
            "What applications does {title} have?",
            "In which areas does {title} play a role?",
            "Where is {title} used?",
            "What practical significance does {title} have?"
        ]
    },
    {
        "type": "technical_specs",
        "templates": [
            "What technical data does {title} have?",
            "List all technical specifications of {title}.",
            "What metrics and parameters does {title} have?",
            "List the technical details of {title}.",
            "What are the technical properties of {title}?"
        ]
    },
    {
        "type": "complete_overview",
        "templates": [
            "Give a complete overview of {title}.",
            "Summarize all important information about {title}.",
            "What are all the relevant details about {title}?",
            "Describe {title} with all available information.",
            "Explain {title} comprehensively with all details."
        ]
    }
]

# ========================================
# DEVELOPER INSTRUCTIONS
# ========================================

DEVELOPER_INSTRUCTIONS = {
    "de": [
        "Du bist ein hilfreicher technischer Assistent mit detailliertem Fachwissen.",
        "Du bist ein KI-Chatbot mit einer lebhaften und energischen Persönlichkeit.",
        "Du bist ein intelligenter Assistent, der Kundenservice-Anfragen beantworten kann.",
        "Du bist ein Fachexperte, der gründlich nachdenkt bevor er antwortet.",
        "Du bist ein analytischer Assistent, der Probleme methodisch durchdenkt.",
        "Du bist ein Bildungsassistent, der Wissen strukturiert vermittelt."
    ],
    "en": [
        "You are a helpful technical assistant with detailed expertise.",
        "You are an AI chatbot with a lively and energetic personality.",
        "You are an intelligent assistant that can answer customer service queries.",
        "You are a domain expert who thinks thoroughly before responding.",
        "You are an analytical assistant that thinks through problems methodically.",
        "You are an educational assistant that conveys knowledge in a structured manner."
    ]
}

# ========================================
# API CONNECTION
# ========================================

def check_api_connection():
    """Checks API connection (OpenAI or Ollama)."""
    if USE_OPENAI_API:
        return check_openai_connection()
    else:
        return check_ollama_connection()

def check_openai_connection():
    """Checks OpenAI API connection."""
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
            print(f"✅ OpenAI API connection successful ({OPENAI_BASE_URL})")
            print(f"✅ Model '{OPENAI_MODEL_NAME}' is available")
            return True
        else:
            print(f"❌ OpenAI API not reachable (Status: {response.status_code})")
            return False
            
    except requests.RequestException as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False

def check_ollama_connection():
    """Checks Ollama API connection."""
    try:
        headers = {
            'Authorization': f'Bearer {OLLAMA_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(f"{OLLAMA_SERVER_URL}/api/tags", headers=headers, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            print(f"✅ Ollama connection successful ({OLLAMA_SERVER_URL})")
            
            if OLLAMA_MODEL_NAME in model_names:
                print(f"✅ Model '{OLLAMA_MODEL_NAME}' is available")
                return True
            else:
                print(f"❌ Model '{OLLAMA_MODEL_NAME}' not found!")
                return False
        else:
            print(f"❌ Ollama not reachable (Status: {response.status_code})")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

# ========================================
# FILE PROCESSING
# ========================================

def load_lexikon_files(input_dir: str) -> List[Dict[str, Any]]:
    """Loads all lexicon JSON files from the input directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory '{input_dir}' does not exist!")
        return []
    
    lexikon_patterns = ["lexikon_*.json", "processed_*.json", "*_lexikon.json"]
    lexikon_files = []
    
    for pattern in lexikon_patterns:
        lexikon_files.extend(list(input_path.glob(pattern)))
    
    lexikon_files = list(set(lexikon_files))
    
    print(f"📁 Found lexicon files: {len(lexikon_files)}")
    
    all_entries = []
    
    for file in lexikon_files:
        print(f"📖 Loading: {file.name}")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if 'lexikon_entries' in data:
                    entries = data.get('lexikon_entries', [])
                elif 'entries' in data:
                    entries = data.get('entries', [])
                elif 'data' in data:
                    entries = data.get('data', [])
                else:
                    if isinstance(data, list):
                        entries = data
                    else:
                        entries = []
                
                print(f"   - {len(entries)} entries found")
                all_entries.extend(entries)
                
        except Exception as e:
            print(f"❌ Error loading {file.name}: {e}")
    
    print(f"📊 Total: {len(all_entries)} lexicon entries")
    return all_entries

# ========================================
# LANGUAGE DETECTION & CONTENT ANALYSIS
# ========================================

def detect_language(text: str) -> str:
    """Detects the language of the text."""
    german_indicators = ['der', 'die', 'das', 'und', 'oder', 'ist', 'sind', 'von', 'mit', 'für']
    english_indicators = ['the', 'and', 'or', 'is', 'are', 'of', 'with', 'for', 'on', 'in']
    
    text_lower = text.lower()
    german_count = sum(1 for word in german_indicators if f' {word} ' in f' {text_lower} ')
    english_count = sum(1 for word in english_indicators if f' {word} ' in f' {text_lower} ')
    
    if german_count > english_count:
        return "German"
    elif english_count > german_count:
        return "English"
    else:
        return "German"  # Default

def detect_content_features(lexikon_entry: str) -> Dict[str, bool]:
    """Detects special content features in the lexicon entry."""
    features = {
        'has_technical_data': any(keyword in lexikon_entry.lower() for keyword in 
                                 ['temperatur', 'messbereich', 'spezifikationen', 'spannung', 
                                  'frequenz', 'genauigkeit', 'temperature', 'range', 
                                  'specifications', 'voltage', 'frequency', 'accuracy']),
        'has_numbers': any(char.isdigit() for char in lexikon_entry),
        'has_list': any(marker in lexikon_entry for marker in ['•', '●', '- ', '* ', '1.', '2.']),
        'has_product_code': bool(re.search(r'\b[A-Z]{2,4}-\d{2,4}\b', lexikon_entry)) if 're' in globals() else False
    }
    return features

# ========================================
# THINKING/ANALYSIS GENERATION
# ========================================

def generate_analysis_content(title: str, question: str, lexikon_entry: str, language: str, question_type: str) -> str:
    """Generates analysis/thinking content for the dataset."""
    features = detect_content_features(lexikon_entry)
    
    if language == "German":
        analysis_parts = []
        
        # Opening analysis
        analysis_parts.append(f"Der Benutzer fragt nach {title}.")
        
        # Analyze what needs to be done
        if question_type == "definition":
            analysis_parts.append(f"Ich muss eine klare Definition von {title} liefern und die Hauptmerkmale erklären.")
        elif question_type == "technical_specs":
            analysis_parts.append(f"Ich werde alle technischen Spezifikationen von {title} auflisten.")
        elif question_type == "application":
            analysis_parts.append(f"Ich erkläre die praktischen Anwendungen von {title}.")
        elif question_type == "complete_overview":
            analysis_parts.append(f"Ich werde eine umfassende Übersicht über {title} geben, die alle Aspekte abdeckt.")
        else:
            analysis_parts.append(f"Ich werde eine detaillierte Antwort über {title} geben.")
        
        # Note special features
        if features['has_technical_data']:
            analysis_parts.append("Ich sehe technische Spezifikationen, die ich genau aufführen muss.")
        if features['has_numbers']:
            analysis_parts.append("Es gibt wichtige numerische Daten, die exakt übernommen werden müssen.")
        if features['has_list']:
            analysis_parts.append("Die Information enthält Listen, die ich strukturiert wiedergeben werde.")
        
        # Closing
        analysis_parts.append("Ich werde nun eine vollständige und strukturierte Antwort formulieren.")
        
    else:  # English
        analysis_parts = []
        
        # Opening analysis
        analysis_parts.append(f"The user is asking about {title}.")
        
        # Analyze what needs to be done
        if question_type == "definition":
            analysis_parts.append(f"I need to provide a clear definition of {title} and explain its main features.")
        elif question_type == "technical_specs":
            analysis_parts.append(f"I'll list all technical specifications for {title}.")
        elif question_type == "application":
            analysis_parts.append(f"I'll explain the practical applications of {title}.")
        elif question_type == "complete_overview":
            analysis_parts.append(f"I'll provide a comprehensive overview of {title} covering all aspects.")
        else:
            analysis_parts.append(f"I'll provide a detailed answer about {title}.")
        
        # Note special features
        if features['has_technical_data']:
            analysis_parts.append("I see technical specifications that I need to list accurately.")
        if features['has_numbers']:
            analysis_parts.append("There are important numerical data points to preserve exactly.")
        if features['has_list']:
            analysis_parts.append("The information contains lists that I'll present in a structured way.")
        
        # Closing
        analysis_parts.append("I'll now formulate a complete and structured response.")
    
    return " ".join(analysis_parts)

# ========================================
# PROMPT GENERATION
# ========================================

def generate_qa_prompt(title: str, lexikon_entry: str, question_type: Dict[str, Any]) -> str:
    """Generates a prompt for QA creation."""
    
    features = detect_content_features(lexikon_entry)
    language = detect_language(lexikon_entry)
    
    special_instructions = []
    
    if features['has_technical_data']:
        special_instructions.append("- Include ALL technical specifications and values exactly")
    
    if features['has_numbers']:
        special_instructions.append("- Preserve ALL numbers and measurements exactly")
    
    if features['has_list']:
        special_instructions.append("- Maintain all lists and enumerations")
    
    if features['has_product_code']:
        special_instructions.append("- Keep all product codes and names exactly as given")
    
    special_instructions_text = "\n".join(special_instructions) if special_instructions else ""
    
    question_template = random.choice(question_type["templates"])
    
    prompt = f"""
Create a Q&A pair for a thinking/reasoning dataset.

TOPIC: {title}
QUESTION TYPE: {question_type["type"]}
QUESTION TEMPLATE: {question_template}
LANGUAGE: {language}

SOURCE CONTENT:
{lexikon_entry}

REQUIREMENTS:
1. Create a natural question based on the template
2. Provide a complete answer with ALL information from the source
3. Use the same language as detected ({language})
4. Include every fact, detail, and specification
5. Use markdown formatting for structure

SPECIAL REQUIREMENTS:
{special_instructions_text}

OUTPUT FORMAT:
{{
    "question": "The formulated question",
    "answer": "The complete answer with all details"
}}

IMPORTANT: Respond ONLY with the JSON object!
"""
    return prompt

# ========================================
# API COMMUNICATION
# ========================================

def submit_to_api(prompt: str, retries: int = 3) -> Optional[str]:
    """Sends a request to the chosen API."""
    if USE_OPENAI_API:
        return submit_to_openai_api(prompt, retries)
    else:
        return submit_to_ollama_api(prompt, retries)

def submit_to_openai_api(prompt: str, retries: int = 3) -> Optional[str]:
    """Sends a request to the OpenAI API."""
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": OPENAI_MODEL_NAME,
        "messages": [
            {
                "role": "system", 
                "content": "You are an expert at creating Q&A pairs. Preserve all details accurately."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2500,
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
                print(f"❌ API error {response.status_code}")
                
        except requests.RequestException as e:
            print(f"❌ Request error (attempt {attempt + 1}): {e}")
    
    return None

def submit_to_ollama_api(prompt: str, retries: int = 3) -> Optional[str]:
    """Sends a request to the Ollama API."""
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
                "content": "You are an expert at creating Q&A pairs. Preserve all details accurately."
            },
            {"role": "user", "content": prompt}
        ],
        "options": {
            "num_predict": 2500
        }
    }

    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "").strip()
                return content
            else:
                print(f"❌ API error {response.status_code}")
                
        except requests.RequestException as e:
            print(f"❌ Request error (attempt {attempt + 1}): {e}")
    
    return None

# ========================================
# QA EXTRACTION & GENERATION
# ========================================

def extract_qa_from_response(response: str) -> Optional[Dict[str, str]]:
    """Extracts QA pair from API response."""
    try:
        response = response.strip()
        
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        
        qa_data = json.loads(response)
        
        if "question" in qa_data and "answer" in qa_data:
            return {
                "question": qa_data["question"].strip(),
                "answer": qa_data["answer"].strip()
            }
        else:
            print(f"❌ Invalid format: Missing required fields")
            return None
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def generate_qa_for_entry(entry: Dict[str, Any], num_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generates multiple QA pairs for a lexicon entry."""
    
    title = entry.get('title', entry.get('name', ''))
    lexikon_entry = entry.get('lexikon_entry', entry.get('entry', entry.get('content', '')))
    
    if not title or not lexikon_entry:
        print(f"⚠️ Incomplete entry skipped: {title or 'Unknown'}")
        return []
    
    language = detect_language(lexikon_entry)
    
    if num_questions is None:
        content_length = len(lexikon_entry)
        if content_length < 500:
            num_questions = 2
        elif content_length < 1500:
            num_questions = 3
        else:
            num_questions = 4
    
    print(f"🔄 Generating {num_questions} QA pairs for: {title}")
    
    qa_pairs = []
    used_question_types = []
    
    # Prioritize complete_overview
    question_types_order = ["complete_overview", "technical_specs", "definition", "application", "characteristics", "detailed_explanation"]
    
    for i in range(num_questions):
        # Select question type
        available_types = [qt for qt in QUESTION_TYPES if qt["type"] not in used_question_types]
        if not available_types:
            available_types = QUESTION_TYPES
            used_question_types = []
        
        # Try to use prioritized order
        selected_type = None
        for prio_type in question_types_order:
            type_obj = next((qt for qt in available_types if qt["type"] == prio_type), None)
            if type_obj:
                selected_type = type_obj
                break
        
        if not selected_type:
            selected_type = random.choice(available_types)
        
        used_question_types.append(selected_type["type"])
        
        print(f"   🎯 Creating {selected_type['type']} question...")
        prompt = generate_qa_prompt(title, lexikon_entry, selected_type)
        
        for attempt in range(2):
            response = submit_to_api(prompt)
            if response:
                qa_pair = extract_qa_from_response(response)
                if qa_pair:
                    qa_pair["source"] = entry.get('source', '')
                    qa_pair["title"] = title
                    qa_pair["question_type"] = selected_type["type"]
                    qa_pair["language"] = language
                    qa_pairs.append(qa_pair)
                    print(f"   ✅ QA {len(qa_pairs)}/{num_questions} created")
                    break
    
    return qa_pairs

# ========================================
# THINKING FORMAT CONVERSION
# ========================================

def convert_to_thinking_format(qa_pair: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a QA pair to the thinking dataset format."""
    
    language = qa_pair.get("language", "English")
    lang_code = "de" if language == "German" else "en"
    
    # Select developer instruction
    developer_instruction = random.choice(DEVELOPER_INSTRUCTIONS[lang_code])
    
    # Generate analysis content
    analysis = generate_analysis_content(
        qa_pair["title"],
        qa_pair["question"],
        qa_pair.get("source", ""),
        language,
        qa_pair.get("question_type", "general")
    )
    
    # Create messages array
    messages = [
        {
            "content": f"reasoning language: {language}\n\n{developer_instruction}",
            "role": "system",
            "thinking": None
        },
        {
            "content": qa_pair["question"],
            "role": "user",
            "thinking": None
        },
        {
            "content": qa_pair["answer"],
            "role": "assistant",
            "thinking": analysis
        }
    ]
    
    # Create the complete entry
    thinking_entry = {
        "reasoning_language": language,
        "developer": developer_instruction,
        "user": qa_pair["question"],
        "analysis": analysis,
        "final": qa_pair["answer"],
        "messages": messages
    }
    
    return thinking_entry

# ========================================
# MAIN PROCESSING
# ========================================

def process_lexikon_to_thinking_dataset(input_dir: str, output_dir: str, output_filename: str):
    """Main function: Converts lexicon to thinking dataset format."""
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🧠 Starting conversion Lexicon → Thinking Dataset")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📄 Output file: {output_filename}")
    print(f"🔧 API: {api_name} ({server_url})")
    print(f"🔧 Model: {model_name}")
    print(f"✨ Features:")
    print(f"   - Thinking/Analysis generation: ✅")
    print(f"   - Multilingual support: ✅")
    print(f"   - Complete fact preservation: ✅")
    print(f"   - Markdown formatting: ✅")
    
    # Check API connection
    if not check_api_connection():
        print("❌ API connection failed. Aborting.")
        return
    
    # Load lexicon entries
    lexikon_entries = load_lexikon_files(input_dir)
    
    if not lexikon_entries:
        print("❌ No lexicon entries found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing {len(lexikon_entries)} lexicon entries")
    print(f"{'='*60}")
    
    all_thinking_entries = []
    statistics = {
        'total_entries': len(lexikon_entries),
        'processed_entries': 0,
        'failed_entries': 0,
        'total_qa_pairs': 0,
        'languages': {'German': 0, 'English': 0}
    }
    
    # Process each entry
    for idx, entry in enumerate(lexikon_entries, 1):
        title = entry.get('title', entry.get('name', f'Entry {idx}'))
        print(f"\n🔄 Entry {idx}/{len(lexikon_entries)}: {title}")
        
        try:
            qa_pairs = generate_qa_for_entry(entry)
            
            if qa_pairs:
                # Convert each QA pair to thinking format
                for qa in qa_pairs:
                    thinking_entry = convert_to_thinking_format(qa)
                    all_thinking_entries.append(thinking_entry)
                    
                    lang = qa.get('language', 'English')
                    statistics['languages'][lang] = statistics['languages'].get(lang, 0) + 1
                
                statistics['processed_entries'] += 1
                statistics['total_qa_pairs'] += len(qa_pairs)
                print(f"   📊 {len(qa_pairs)} entries created")
            else:
                statistics['failed_entries'] += 1
                print(f"   ⚠️ No entries generated")
                
        except Exception as e:
            statistics['failed_entries'] += 1
            print(f"   ❌ Error: {e}")
    
    # Save as JSONL (one entry per line)
    output_path = Path(output_dir) / output_filename
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in all_thinking_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Output statistics
    print(f"\n{'='*60}")
    print(f"🎉 Thinking Dataset successfully created!")
    print(f"{'='*60}")
    print(f"📊 Statistics:")
    print(f"   📁 Processing:")
    print(f"      - Total lexicon entries: {statistics['total_entries']}")
    print(f"      - Processed: {statistics['processed_entries']}")
    print(f"      - Failed: {statistics['failed_entries']}")
    print(f"   📝 Generated:")
    print(f"      - Total dataset entries: {len(all_thinking_entries)}")
    print(f"   🌍 Languages:")
    for lang, count in statistics['languages'].items():
        if count > 0:
            print(f"      - {lang}: {count} entries")
    print(f"   💾 Output:")
    print(f"      - File: {output_path}")
    print(f"      - Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Show example
    if all_thinking_entries:
        print(f"\n📋 Example entry:")
        example = all_thinking_entries[0]
        print(f"   Reasoning Language: {example['reasoning_language']}")
        print(f"   Developer: {example['developer'][:50]}...")
        print(f"   User: {example['user'][:100]}...")
        print(f"   Analysis: {example['analysis'][:150]}...")
        print(f"   Final: {example['final'][:150]}...")
        print(f"   Messages: {len(example['messages'])} entries")

# ========================================
# MAIN PROGRAM
# ========================================

if __name__ == "__main__":
    import re
    
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🧠 THINKING DATASET GENERATOR")
    print(f"{'='*60}")
    print(f"🔧 Configuration:")
    print(f"   - API: {api_name}")
    print(f"   - Server: {server_url}")
    print(f"   - Model: {model_name}")
    print(f"   - Output format: Thinking Dataset (JSONL)")
    print(f"   - Features: Analysis/Reasoning with structured format")
    
    # Start processing
    process_lexikon_to_thinking_dataset(INPUT_DIR, OUTPUT_DIR, OUTPUT_FILENAME)