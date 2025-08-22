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
OPENAI_MODEL_NAME = config.get("openai_model_name", "gemma3:12b-it-qat")
OLLAMA_SERVER_URL = config.get("ollama_server_url", "http://localhost:11434")
OLLAMA_API_KEY = config.get("ollama_api_key", "ollama")
OLLAMA_MODEL_NAME = config.get("ollama_model_name", "gemma3:12b-it-qat")
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_SERVER_URL}/api/chat"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_SERVER_URL}/api/tags"

# Directories
INPUT_DIR = "INPUT"
OUTPUT_DIR = "OUTPUT"
OUTPUT_FILENAME = "multilingual_thinking_dataset.json"
OUTPUT_FILENAME_JSONL = "multilingual_thinking_dataset.jsonl"

# Show loaded configuration
print("=" * 60)
print("🧠 CONFIGURATION LOADED (Multilingual-Thinking Format)")
print("=" * 60)
config_loader.print_config_summary()
print("=" * 60)

# ========================================
# EXTENDED QUESTION TYPES FOR BETTER COVERAGE
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
        ],
        "thinking_prompts": [
            "I need to provide a clear definition of {title}. Let me consider its key characteristics and main purpose.",
            "The user wants to understand {title}. I should explain what it is, its main features, and its applications.",
            "Let me think about how to best explain {title} in a comprehensive way."
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
        ],
        "thinking_prompts": [
            "I need to provide a comprehensive explanation of {title}. Let me organize the information systematically.",
            "The user wants detailed information about {title}. I'll cover all important aspects including technical specifications and applications.",
            "Let me structure a detailed response about {title}, ensuring I include all relevant information."
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
        ],
        "thinking_prompts": [
            "I need to identify and explain the key characteristics of {title}. Let me list its main properties.",
            "The user is asking about the specific features of {title}. I should highlight what makes it unique.",
            "Let me think about the distinguishing characteristics and properties of {title}."
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
        ],
        "thinking_prompts": [
            "I need to explain the practical applications of {title}. Let me consider its various use cases.",
            "The user wants to know where and how {title} is used. I'll list its main applications and areas of use.",
            "Let me think about the different contexts where {title} is applied and its practical significance."
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
        ],
        "thinking_prompts": [
            "I need to provide the technical specifications of {title}. Let me list all the relevant parameters and metrics.",
            "The user wants technical details about {title}. I should include all specifications, measurements, and technical data.",
            "Let me organize the technical specifications of {title} in a clear and comprehensive manner."
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
        ],
        "thinking_prompts": [
            "I need to provide a complete overview of {title}. Let me ensure I cover all aspects: definition, characteristics, technical specs, and applications.",
            "The user wants comprehensive information about {title}. I'll structure this to include everything from basic definition to detailed specifications.",
            "Let me create a thorough overview of {title}, making sure no important information is left out."
        ]
    },
    {
        "type": "structure_and_components",
        "templates": [
            "What components does {title} consist of?",
            "How is {title} structured?",
            "What parts does {title} have?",
            "Describe the structure of {title}.",
            "What elements belong to {title}?"
        ],
        "thinking_prompts": [
            "I need to explain the structure and components of {title}. Let me break down its constituent parts.",
            "The user wants to understand how {title} is composed. I'll describe its structure and individual elements.",
            "Let me analyze the components and structure of {title} systematically."
        ]
    }
]

# ========================================
# MULTILINGUAL THINKING SYSTEM PROMPTS
# ========================================

SYSTEM_PROMPTS_THINKING = {
    "technical_de": "Du bist ein hilfreicher technischer Assistent, der Schritt für Schritt denkt und komplexe technische Konzepte klar erklärt. Verwende <thinking> Tags für deinen Denkprozess.",
    "technical_en": "You are a helpful technical assistant that thinks step by step and explains complex technical concepts clearly. Use <thinking> tags for your reasoning process.",
    "educational_de": "Du bist ein hilfreicher Bildungsassistent, der systematisch denkt und Wissen strukturiert vermittelt. Zeige deinen Denkprozess in <thinking> Tags.",
    "educational_en": "You are a helpful educational assistant that thinks systematically and conveys knowledge in a structured manner. Show your thinking process in <thinking> tags.",
    "analytical_de": "Du bist ein analytischer Assistent, der Probleme methodisch durchdenkt. Dokumentiere deine Überlegungen in <thinking> Tags bevor du antwortest.",
    "analytical_en": "You are an analytical assistant that thinks through problems methodically. Document your reasoning in <thinking> tags before answering.",
    "expert_de": "Du bist ein Fachexperte, der gründlich nachdenkt bevor er antwortet. Nutze <thinking> Tags um deine Gedankengänge zu zeigen.",
    "expert_en": "You are a domain expert who thinks thoroughly before responding. Use <thinking> tags to show your thought process."
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
    
    return "de" if german_count > english_count else "en"

def detect_content_features(lexikon_entry: str) -> Dict[str, bool]:
    """Detects special content features in the lexicon entry."""
    features = {
        'has_table': '|' in lexikon_entry or '\t' in lexikon_entry,
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
# THINKING PROCESS GENERATION
# ========================================

def generate_thinking_content(title: str, question_type: Dict[str, Any], lexikon_entry: str, language: str) -> str:
    """Generates thinking/reasoning content for Chain-of-Thought."""
    features = detect_content_features(lexikon_entry)
    
    # Get base thinking prompt
    thinking_template = random.choice(question_type.get("thinking_prompts", []))
    thinking_base = thinking_template.replace("{title}", title)
    
    # Add language-specific thinking elements
    if language == "de":
        thinking_parts = [thinking_base]
        
        if features['has_technical_data']:
            thinking_parts.append("Ich sehe, dass technische Spezifikationen vorhanden sind. Ich werde alle Werte und Parameter genau aufführen.")
        
        if features['has_numbers']:
            thinking_parts.append("Es gibt wichtige numerische Daten, die ich exakt übernehmen muss.")
        
        if features['has_list']:
            thinking_parts.append("Die Information enthält Listen oder Aufzählungen, die ich strukturiert wiedergeben werde.")
        
        if features['has_product_code']:
            thinking_parts.append(f"Dies ist ein spezifisches Produkt mit der Bezeichnung {title}. Ich werde alle produktspezifischen Details einbeziehen.")
        
        thinking_parts.append("Ich werde nun eine vollständige und strukturierte Antwort formulieren.")
        
    else:  # English
        thinking_parts = [thinking_base]
        
        if features['has_technical_data']:
            thinking_parts.append("I see there are technical specifications present. I'll list all values and parameters accurately.")
        
        if features['has_numbers']:
            thinking_parts.append("There are important numerical data points that I need to preserve exactly.")
        
        if features['has_list']:
            thinking_parts.append("The information contains lists or enumerations that I'll present in a structured way.")
        
        if features['has_product_code']:
            thinking_parts.append(f"This is a specific product designated as {title}. I'll include all product-specific details.")
        
        thinking_parts.append("I'll now formulate a complete and structured response.")
    
    return "\n".join(thinking_parts)

# ========================================
# PROMPT GENERATION
# ========================================

def generate_qa_prompt(title: str, lexikon_entry: str, question_type: Dict[str, Any]) -> str:
    """Generates a prompt for QA creation with thinking process."""
    
    features = detect_content_features(lexikon_entry)
    language = detect_language(lexikon_entry)
    
    special_instructions = []
    
    if features['has_table']:
        special_instructions.append("- Preserve ALL tables and structured data completely")
    
    if features['has_technical_data']:
        special_instructions.append("- Include ALL technical specifications and values exactly")
    
    if features['has_numbers']:
        special_instructions.append("- Preserve ALL numbers, measurements, and quantitative data exactly")
    
    if features['has_list']:
        special_instructions.append("- Maintain all lists and enumerations")
    
    if features['has_product_code']:
        special_instructions.append("- Treat as specific product, maintain all product codes and names")
    
    special_instructions_text = "\n".join(special_instructions) if special_instructions else ""
    
    question_template = random.choice(question_type["templates"])
    
    prompt = f"""
Create a high-quality Q&A pair with Chain-of-Thought reasoning for the Multilingual-Thinking dataset.

TOPIC: {title}
QUESTION TYPE: {question_type["type"]}
QUESTION TEMPLATE: {question_template}
DETECTED LANGUAGE: {language.upper()}

LEXICON ENTRY (SOURCE):
{lexikon_entry}

TASK:
1. Create a natural question based on the template
2. Generate a thinking process that shows step-by-step reasoning
3. Provide a complete answer after the thinking process
4. Include ALL information from the lexicon entry
5. Use the same language as the source content

CRITICAL REQUIREMENTS:
- NO information loss - preserve every fact and detail
- The thinking process should be logical and show how you organize the information
- Use Markdown formatting in the answer (bold, italics, lists, tables)
- Technical specifications must be complete and accurate

SPECIAL CONTENT REQUIREMENTS:
{special_instructions_text}

OUTPUT FORMAT:
{{
    "question": "The formulated question",
    "thinking": "The step-by-step thinking process",
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
                "content": "You are an expert at creating Q&A pairs with Chain-of-Thought reasoning. You preserve all details and use thinking tags."
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
                "content": "You are an expert at creating Q&A pairs with Chain-of-Thought reasoning. You preserve all details and use thinking tags."
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
    """Extracts QA pair with thinking from API response."""
    try:
        response = response.strip()
        
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        
        qa_data = json.loads(response)
        
        if "question" in qa_data and "answer" in qa_data:
            # If thinking is missing, generate a simple one
            if "thinking" not in qa_data:
                qa_data["thinking"] = "Let me analyze the information and provide a comprehensive answer."
            
            return {
                "question": qa_data["question"].strip(),
                "thinking": qa_data["thinking"].strip(),
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
    """Generates multiple QA pairs with thinking for a lexicon entry."""
    
    title = entry.get('title', entry.get('name', ''))
    lexikon_entry = entry.get('lexikon_entry', entry.get('entry', entry.get('content', '')))
    
    if not title or not lexikon_entry:
        print(f"⚠️ Incomplete entry skipped: {title or 'Unknown'}")
        return []
    
    language = detect_language(lexikon_entry)
    
    if num_questions is None:
        content_length = len(lexikon_entry)
        if content_length < 500:
            num_questions = random.randint(2, 4)
        elif content_length < 1500:
            num_questions = random.randint(3, 5)
        else:
            num_questions = random.randint(4, 6)
    
    print(f"🔄 Generating {num_questions} QA pairs for: {title}")
    
    qa_pairs = []
    used_question_types = set()
    
    # Ensure complete_overview is first
    overview_type = next((qt for qt in QUESTION_TYPES if qt["type"] == "complete_overview"), None)
    
    if overview_type:
        print(f"   🎯 Creating complete overview...")
        prompt = generate_qa_prompt(title, lexikon_entry, overview_type)
        
        for attempt in range(3):
            response = submit_to_api(prompt)
            if response:
                qa_pair = extract_qa_from_response(response)
                if qa_pair:
                    # Add metadata
                    qa_pair["source"] = entry.get('source', '')
                    qa_pair["title"] = title
                    qa_pair["question_type"] = "complete_overview"
                    qa_pair["language"] = language
                    
                    # Generate thinking if not present
                    if not qa_pair.get("thinking"):
                        qa_pair["thinking"] = generate_thinking_content(
                            title, overview_type, lexikon_entry, language
                        )
                    
                    qa_pairs.append(qa_pair)
                    used_question_types.add("complete_overview")
                    print(f"   ✅ QA 1/{num_questions} created (complete_overview)")
                    break
    
    # Generate additional questions
    remaining_questions = num_questions - len(qa_pairs)
    
    for i in range(remaining_questions):
        available_types = [qt for qt in QUESTION_TYPES if qt["type"] not in used_question_types]
        
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
                    qa_pair["language"] = language
                    
                    # Generate thinking if not present
                    if not qa_pair.get("thinking"):
                        qa_pair["thinking"] = generate_thinking_content(
                            title, question_type, lexikon_entry, language
                        )
                    
                    qa_pairs.append(qa_pair)
                    print(f"   ✅ QA {len(qa_pairs)}/{num_questions} created ({question_type['type']})")
                    break
    
    return qa_pairs

# ========================================
# MULTILINGUAL THINKING FORMAT CONVERSION
# ========================================

def convert_to_multilingual_thinking_format(qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts QA pairs to HuggingFace Multilingual-Thinking format."""
    thinking_dataset = []
    
    for qa in qa_pairs:
        language = qa.get("language", "en")
        
        # Select appropriate system prompt based on language and type
        if language == "de":
            system_prompts = [k for k in SYSTEM_PROMPTS_THINKING.keys() if k.endswith("_de")]
        else:
            system_prompts = [k for k in SYSTEM_PROMPTS_THINKING.keys() if k.endswith("_en")]
        
        system_key = random.choice(system_prompts)
        system_prompt = SYSTEM_PROMPTS_THINKING[system_key]
        
        # Create the thinking-enhanced answer
        thinking_answer = f"<thinking>\n{qa['thinking']}\n</thinking>\n\n{qa['answer']}"
        
        # Create messages array
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": qa["question"]
            },
            {
                "role": "assistant",
                "content": thinking_answer
            }
        ]
        
        # Create the entry
        thinking_entry = {
            "messages": messages,
            "metadata": {
                "source": qa.get("source", ""),
                "title": qa.get("title", ""),
                "question_type": qa.get("question_type", ""),
                "language": language,
                "has_thinking": True,
                "answer_length": len(qa["answer"]),
                "thinking_length": len(qa["thinking"]),
                "total_length": len(thinking_answer)
            }
        }
        
        thinking_dataset.append(thinking_entry)
    
    return thinking_dataset

def create_multi_turn_thinking_conversations(qa_pairs: List[Dict[str, Any]], max_turns: int = 3) -> List[Dict[str, Any]]:
    """Creates multi-turn conversations with thinking for each response."""
    multi_turn_dataset = []
    
    # Group by title
    grouped_qa = {}
    for qa in qa_pairs:
        title = qa.get("title", "Unknown")
        if title not in grouped_qa:
            grouped_qa[title] = []
        grouped_qa[title].append(qa)
    
    for title, qa_group in grouped_qa.items():
        if len(qa_group) >= 2:
            num_turns = min(len(qa_group), max_turns)
            selected_qa = random.sample(qa_group, num_turns)
            
            language = selected_qa[0].get("language", "en")
            
            # Select system prompt
            if language == "de":
                system_prompts = [k for k in SYSTEM_PROMPTS_THINKING.keys() if k.endswith("_de")]
            else:
                system_prompts = [k for k in SYSTEM_PROMPTS_THINKING.keys() if k.endswith("_en")]
            
            system_key = random.choice(system_prompts)
            system_prompt = SYSTEM_PROMPTS_THINKING[system_key]
            
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
            
            for i, qa in enumerate(selected_qa):
                # Add user question
                if i == 0:
                    question = qa["question"]
                else:
                    # Follow-up questions
                    if language == "de":
                        follow_ups = ["Kannst du auch erklären,", "Was ist mit", "Erzähl mir mehr über"]
                    else:
                        follow_ups = ["Can you also explain", "What about", "Tell me more about"]
                    follow_up = random.choice(follow_ups)
                    question = f"{follow_up} {qa['question'].lower()}"
                
                messages.append({
                    "role": "user",
                    "content": question
                })
                
                # Add assistant response with thinking
                thinking_answer = f"<thinking>\n{qa['thinking']}\n</thinking>\n\n{qa['answer']}"
                messages.append({
                    "role": "assistant",
                    "content": thinking_answer
                })
            
            # Create multi-turn entry
            multi_turn_entry = {
                "messages": messages,
                "metadata": {
                    "source": selected_qa[0].get("source", ""),
                    "title": title,
                    "question_types": [qa.get("question_type", "") for qa in selected_qa],
                    "language": language,
                    "is_multi_turn": True,
                    "num_turns": num_turns,
                    "total_length": sum(len(msg["content"]) for msg in messages)
                }
            }
            
            multi_turn_dataset.append(multi_turn_entry)
    
    return multi_turn_dataset

# ========================================
# MAIN PROCESSING
# ========================================

def process_lexikon_to_thinking_dataset(input_dir: str, output_dir: str, output_filename: str):
    """Main function: Converts lexicon to Multilingual-Thinking dataset."""
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🧠 Starting conversion Lexicon → Multilingual-Thinking Dataset")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📄 Output file (JSON): {output_filename}")
    print(f"📄 Output file (JSONL): {OUTPUT_FILENAME_JSONL}")
    print(f"🔧 API: {api_name} ({server_url})")
    print(f"🔧 Model: {model_name}")
    print(f"✨ Features:")
    print(f"   - Chain-of-Thought reasoning: ✅")
    print(f"   - <thinking> tags: ✅")
    print(f"   - Multilingual support: ✅")
    print(f"   - Multi-turn conversations: ✅")
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
    
    all_qa_pairs = []
    statistics = {
        'total_entries': len(lexikon_entries),
        'processed_entries': 0,
        'failed_entries': 0,
        'total_qa_pairs': 0,
        'qa_by_type': {},
        'qa_by_language': {'de': 0, 'en': 0}
    }
    
    # Process each entry
    for idx, entry in enumerate(lexikon_entries, 1):
        title = entry.get('title', entry.get('name', f'Entry {idx}'))
        print(f"\n🔄 Entry {idx}/{len(lexikon_entries)}: {title}")
        
        try:
            qa_pairs = generate_qa_for_entry(entry)
            
            if qa_pairs:
                all_qa_pairs.extend(qa_pairs)
                statistics['processed_entries'] += 1
                statistics['total_qa_pairs'] += len(qa_pairs)
                
                for qa in qa_pairs:
                    q_type = qa.get('question_type', 'unknown')
                    statistics['qa_by_type'][q_type] = statistics['qa_by_type'].get(q_type, 0) + 1
                    
                    lang = qa.get('language', 'en')
                    statistics['qa_by_language'][lang] = statistics['qa_by_language'].get(lang, 0) + 1
                
                print(f"   📊 {len(qa_pairs)} QA pairs generated")
            else:
                statistics['failed_entries'] += 1
                print(f"   ⚠️ No QA pairs generated")
                
        except Exception as e:
            statistics['failed_entries'] += 1
            print(f"   ❌ Error: {e}")
    
    # Convert to Multilingual-Thinking format
    print(f"\n🔄 Converting to Multilingual-Thinking format...")
    thinking_dataset = convert_to_multilingual_thinking_format(all_qa_pairs)
    
    # Create multi-turn conversations
    print(f"🔄 Creating multi-turn conversations...")
    multi_turn_conversations = create_multi_turn_thinking_conversations(all_qa_pairs, max_turns=3)
    
    # Combine all conversations
    all_conversations = thinking_dataset + multi_turn_conversations
    random.shuffle(all_conversations)
    
    # Calculate statistics
    total_conversations = len(all_conversations)
    multi_turn_count = sum(1 for conv in all_conversations if conv['metadata'].get('is_multi_turn', False))
    avg_thinking_length = sum(
        conv['metadata'].get('thinking_length', 0) 
        for conv in all_conversations if 'thinking_length' in conv['metadata']
    ) / len([c for c in all_conversations if 'thinking_length' in c['metadata']]) if thinking_dataset else 0
    
    # Save as JSON
    output_path = Path(output_dir) / output_filename
    final_dataset = {
        "metadata": {
            "format": "multilingual_thinking",
            "version": "1.0",
            "compatible_with": "HuggingFaceH4/Multilingual-Thinking",
            "total_lexicon_entries": statistics['total_entries'],
            "processed_entries": statistics['processed_entries'],
            "failed_entries": statistics['failed_entries'],
            "total_qa_pairs_generated": statistics['total_qa_pairs'],
            "total_conversations": total_conversations,
            "single_turn_conversations": total_conversations - multi_turn_count,
            "multi_turn_conversations": multi_turn_count,
            "avg_thinking_length": avg_thinking_length,
            "languages": statistics['qa_by_language'],
            "qa_by_type": statistics['qa_by_type'],
            "api_used": api_name,
            "model_used": model_name
        },
        "conversations": all_conversations
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    
    # Save as JSONL (for direct HuggingFace compatibility)
    jsonl_path = Path(output_dir) / OUTPUT_FILENAME_JSONL
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
    
    # Output statistics
    print(f"\n{'='*60}")
    print(f"🎉 Multilingual-Thinking Dataset successfully created!")
    print(f"{'='*60}")
    print(f"📊 Statistics:")
    print(f"   📁 Processing:")
    print(f"      - Total entries: {statistics['total_entries']}")
    print(f"      - Processed: {statistics['processed_entries']}")
    print(f"      - Failed: {statistics['failed_entries']}")
    print(f"   💬 Conversations:")
    print(f"      - Total: {total_conversations}")
    print(f"      - Single-turn: {total_conversations - multi_turn_count}")
    print(f"      - Multi-turn: {multi_turn_count}")
    print(f"   🧠 Thinking:")
    print(f"      - Avg thinking length: {avg_thinking_length:.0f} chars")
    print(f"   🌍 Languages:")
    for lang, count in statistics['qa_by_language'].items():
        print(f"      - {lang}: {count} QA pairs")
    print(f"   🎯 Question types:")
    for q_type, count in sorted(statistics['qa_by_type'].items(), key=lambda x: x[1], reverse=True):
        print(f"      - {q_type}: {count}")
    print(f"   💾 Output:")
    print(f"      - JSON: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    print(f"      - JSONL: {jsonl_path} ({jsonl_path.stat().st_size / 1024:.1f} KB)")
    
    # Show example
    if all_conversations:
        print(f"\n📋 Example conversation:")
        example = all_conversations[0]
        for i, msg in enumerate(example['messages'][:3]):
            role = msg['role']
            content = msg['content']
            if '<thinking>' in content:
                thinking_end = content.index('</thinking>') + len('</thinking>')
                preview = content[:thinking_end] + "\n[answer follows...]"
            else:
                preview = content[:200] + "..." if len(content) > 200 else content
            print(f"\n   {role.upper()}:")
            print(f"   {preview}")

# ========================================
# MAIN PROGRAM
# ========================================

if __name__ == "__main__":
    import re
    
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🧠 MULTILINGUAL-THINKING DATASET GENERATOR")
    print(f"{'='*60}")
    print(f"🔧 Configuration:")
    print(f"   - API: {api_name}")
    print(f"   - Server: {server_url}")
    print(f"   - Model: {model_name}")
    print(f"   - Format: HuggingFaceH4/Multilingual-Thinking")
    print(f"   - Features: Chain-of-Thought with <thinking> tags")
    
    # Start processing
    process_lexikon_to_thinking_dataset(INPUT_DIR, OUTPUT_DIR, OUTPUT_FILENAME)