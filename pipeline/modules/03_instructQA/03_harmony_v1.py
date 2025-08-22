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
OUTPUT_FILENAME = "harmony_gpt_oss_dataset.json"
OUTPUT_FILENAME_TAGGED = "harmony_gpt_oss_dataset_tagged.txt"  # For GPT-OSS tag format

# Show loaded configuration
print("=" * 60)
print("🎯 CONFIGURATION LOADED (03_instructQA - Harmony Format)")
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
        "type": "context",
        "templates": [
            "In what context is {title} relevant?",
            "Why is {title} important?",
            "What significance does {title} have?",
            "In what connection does {title} stand?",
            "What role does {title} play?"
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
    },
    {
        "type": "data_and_facts",
        "templates": [
            "What concrete data and facts exist about {title}?",
            "List all numbers and measurements for {title}.",
            "What quantitative information is available for {title}?",
            "List all factual information about {title}.",
            "What are the measurable properties of {title}?"
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
        ]
    },
    {
        "type": "comparison_and_differences",
        "templates": [
            "How does {title} differ from similar concepts?",
            "What makes {title} special?",
            "What variants of {title} exist?",
            "How does {title} distinguish itself?",
            "What are the unique features of {title}?"
        ]
    }
]

# ========================================
# HARMONY FORMAT SPECIFIC CONFIGURATIONS
# ========================================

# System prompts for different conversation styles
SYSTEM_PROMPTS = {
    "technical": "You are a knowledgeable technical assistant specializing in industrial sensors and technical products.",
    "educational": "You are a helpful educational assistant that explains complex topics in a clear and structured manner.",
    "professional": "You are a professional consultant providing detailed information about industrial products and solutions.",
    "expert": "You are a domain expert providing comprehensive and accurate information about specialized topics."
}

# Conversation starters for more natural dialogue
CONVERSATION_STARTERS = [
    "I'd like to know more about",
    "Could you tell me about",
    "I'm interested in learning about",
    "Can you provide information on",
    "Please explain",
    "I need details about",
    "What can you tell me about",
    "I'm researching"
]

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
            if response.status_code == 401:
                print("🔒 Authentication failed - check OPENAI_API_KEY")
            elif response.status_code == 404:
                print("❌ Model not found - check OPENAI_MODEL_NAME")
            return False
            
    except requests.RequestException as e:
        print(f"❌ OpenAI API connection failed: {e}")
        print(f"💡 Check: Server running on {OPENAI_BASE_URL}?")
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
            print(f"📋 Available models: {', '.join(model_names[:3])}..." if len(model_names) > 3 else f"📋 Available models: {', '.join(model_names)}")
            
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
    
    # Search for all possible lexicon files
    lexikon_patterns = ["lexikon_*.json", "processed_*.json", "*_lexikon.json"]
    lexikon_files = []
    
    for pattern in lexikon_patterns:
        lexikon_files.extend(list(input_path.glob(pattern)))
    
    # Remove duplicates
    lexikon_files = list(set(lexikon_files))
    
    print(f"📁 Found lexicon files: {len(lexikon_files)}")
    
    all_entries = []
    
    for file in lexikon_files:
        print(f"📖 Loading: {file.name}")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Support different structures
                if 'lexikon_entries' in data:
                    entries = data.get('lexikon_entries', [])
                elif 'entries' in data:
                    entries = data.get('entries', [])
                elif 'data' in data:
                    entries = data.get('data', [])
                else:
                    # Fallback: Try to interpret file as list
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
# PROMPT GENERATION WITH FACT PRESERVATION
# ========================================

def detect_content_features(lexikon_entry: str) -> Dict[str, bool]:
    """Detects special content features in the lexicon entry."""
    features = {
        'has_table': '|' in lexikon_entry or '\t' in lexikon_entry,
        'has_technical_data': any(keyword in lexikon_entry.lower() for keyword in 
                                 ['temperaturbereich', 'messbereich', 'spezifikationen', 
                                  'betriebsspannung', 'frequenz', 'genauigkeit', 'temperature range',
                                  'measurement range', 'specifications', 'operating voltage', 
                                  'frequency', 'accuracy']),
        'has_numbers': any(char.isdigit() for char in lexikon_entry),
        'has_list': any(marker in lexikon_entry for marker in ['•', '●', '○', '■', '- ', '* ', '1.', '2.']),
        'has_product_code': bool(re.search(r'\b[A-Z]{2,4}-\d{2,4}\b', lexikon_entry)) if 're' in globals() else False,
        'is_structured_data': 'bundesland' in lexikon_entry.lower() or 'tabelle' in lexikon_entry.lower() or 'table' in lexikon_entry.lower()
    }
    return features

def detect_language(text: str) -> str:
    """Detects the language of the text (simple detection)."""
    # Simple language detection based on common words
    german_indicators = ['der', 'die', 'das', 'und', 'oder', 'ist', 'sind', 'von', 'mit', 'für', 'auf', 'in', 'zu', 'bei', 'nach', 'über', 'durch', 'unter', 'gegen', 'ohne']
    english_indicators = ['the', 'and', 'or', 'is', 'are', 'of', 'with', 'for', 'on', 'in', 'to', 'at', 'after', 'over', 'by', 'under', 'against', 'without']
    
    text_lower = text.lower()
    german_count = sum(1 for word in german_indicators if word in text_lower)
    english_count = sum(1 for word in english_indicators if word in text_lower)
    
    if german_count > english_count:
        return "German"
    elif english_count > german_count:
        return "English"
    else:
        return "German"  # Default to German for this pipeline

def generate_qa_prompt(title: str, lexikon_entry: str, question_type: Dict[str, Any]) -> str:
    """Generates a prompt for QA creation with complete fact preservation."""
    
    # Detect content features
    features = detect_content_features(lexikon_entry)
    
    # Detect language of the source content
    source_language = detect_language(lexikon_entry)
    
    # Build special instructions based on content features
    special_instructions = []
    
    if features['has_table']:
        special_instructions.append("""
- The lexicon entry contains TABLES or structured data - these must be COMPLETELY preserved
- Use Markdown tables (|---|---|) for tabular data
- Maintain structure and formatting""")
    
    if features['has_technical_data']:
        special_instructions.append("""
- The entry contains TECHNICAL DATA - all specifications, values and parameters must be preserved
- Use **bold text** for technical terms
- Structure technical data as list or table""")
    
    if features['has_numbers']:
        special_instructions.append("""
- ALL numbers, measurements, percentages and quantitative data must be exactly preserved
- Do not round values and do not change units""")
    
    if features['has_list']:
        special_instructions.append("""
- Maintain all enumerations and lists
- Use the same list structure as in the original""")
    
    if features['has_product_code']:
        special_instructions.append("""
- This is a PRODUCT ENTRY - maintain product codes and names exactly
- Treat it as a concrete product, not as a general concept""")
    
    special_instructions_text = "\n".join(special_instructions) if special_instructions else ""
    
    # Choose a random template for the question (but in detected language)
    question_template = random.choice(question_type["templates"])
    
    prompt = f"""
Create a high-quality question-answer pair for a Harmony format dataset (GPT-OSS style).

TOPIC: {title}
QUESTION TYPE: {question_type["type"]}
QUESTION TEMPLATE: {question_template}

LEXICON ENTRY (SOURCE):
{lexikon_entry}

LANGUAGE DETECTION: The source content appears to be in {source_language}.

TASK:
1. Create a natural question based on the template "{question_template}"
2. Answer the question COMPLETELY and THOROUGHLY based on the lexicon entry
3. The answer should be comprehensive (3-10 sentences or more for complex topics)
4. Include ALL relevant information, facts and details from the lexicon entry
5. Style: Professional, educational, like a subject expert
6. CRITICAL: Respond in the same language as the source document. If the source is in German, respond in German. If the source is in English, respond in English.

CRITICAL REQUIREMENTS:
- NO information loss! Every detail, every number, every fact must be preserved
- For technical topics: Include ALL specifications and values
- For products: Maintain product codes and names exactly
- For lists/tables: Preserve structure completely

MARKDOWN FORMATTING:
- Use **bold text** for important terms and headings
- Use *italic text* for emphasis
- Structure with headings (##, ###) when appropriate
- Use lists (- or *) for enumerations
- Use Markdown tables for tabular data
- Code blocks ``` for technical values when appropriate

SPECIAL CONTENT INSTRUCTIONS:
{special_instructions_text}

ANSWER LENGTH:
- Minimum: 3 complete sentences
- Maximum: As long as necessary to convey ALL information
- For complex topics or lots of data: Feel free to use 15-20 sentences

OUTPUT FORMAT:
{{
    "question": "The formulated question based on the template",
    "answer": "The complete, detailed, fact-rich answer with all details from the lexicon entry"
}}

IMPORTANT: Respond ONLY with the JSON object, no additional explanations!
"""
    return prompt

# ========================================
# API COMMUNICATION
# ========================================

def submit_to_api(prompt: str, retries: int = 3) -> Optional[str]:
    """Sends a request to the chosen API and retrieves the response."""
    if USE_OPENAI_API:
        return submit_to_openai_api(prompt, retries)
    else:
        return submit_to_ollama_api(prompt, retries)

def submit_to_openai_api(prompt: str, retries: int = 3) -> Optional[str]:
    """Sends a request to the OpenAI API and retrieves the response."""
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": OPENAI_MODEL_NAME,
        "messages": [
            {
                "role": "system", 
                "content": "You are an expert at creating high-quality question-answer pairs for Harmony format datasets. Your answers are complete, fact-rich and preserve ALL details from the source. You use Markdown formatting for better structure. You respond in the same language as the source document."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,  # Increased for longer, complete answers
        "temperature": 0.3   # Low for factual accuracy
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
                print(f"❌ OpenAI API error {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            print(f"❌ API error (attempt {attempt + 1}): {e}")
    
    return None

def submit_to_ollama_api(prompt: str, retries: int = 3) -> Optional[str]:
    """Sends a request to the Ollama API and retrieves the response."""
    headers = {
        'Authorization': f'Bearer {OLLAMA_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "temperature": 0.3,  # Low for factual accuracy
        "stream": False,
        "messages": [
            {
                "role": "system", 
                "content": "You are an expert at creating high-quality question-answer pairs for Harmony format datasets. Your answers are complete, fact-rich and preserve ALL details from the source. You use Markdown formatting for better structure. You respond in the same language as the source document."
            },
            {"role": "user", "content": prompt}
        ],
        "options": {
            "num_predict": 2000  # Increased for longer answers
        }
    }

    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                content = response.json().get("message", {}).get("content", "").strip()
                return content
            else:
                print(f"❌ Ollama API error {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            print(f"❌ API error (attempt {attempt + 1}): {e}")
    
    return None

# ========================================
# QA GENERATION
# ========================================

def extract_qa_from_response(response: str) -> Optional[Dict[str, str]]:
    """Extracts QA pair from the API response."""
    try:
        # Clean the response
        response = response.strip()
        
        # Remove code block markers if present
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        
        # Parse JSON
        qa_data = json.loads(response)
        
        # Validate structure
        if "question" in qa_data and "answer" in qa_data:
            return {
                "question": qa_data["question"].strip(),
                "answer": qa_data["answer"].strip()
            }
        else:
            print(f"❌ Invalid QA format: Missing fields")
            return None
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print(f"Raw Response (first 200 characters): {response[:200]}...")
        return None
    except Exception as e:
        print(f"❌ Unexpected error during extraction: {e}")
        return None

def generate_qa_for_entry(entry: Dict[str, Any], num_questions: Optional[int] = None) -> List[Dict[str, Any]]:
    """Generates multiple QA pairs for a lexicon entry with focus on completeness."""
    
    # Extract relevant fields (support different structures)
    title = entry.get('title', entry.get('name', ''))
    lexikon_entry = entry.get('lexikon_entry', entry.get('entry', entry.get('content', '')))
    
    if not title or not lexikon_entry:
        print(f"⚠️ Incomplete entry skipped: {title or 'Unknown'}")
        return []
    
    # Determine number of questions based on content length
    if num_questions is None:
        content_length = len(lexikon_entry)
        if content_length < 500:
            num_questions = random.randint(3, 5)
        elif content_length < 1500:
            num_questions = random.randint(4, 7)
        else:
            num_questions = random.randint(5, 8)
    
    print(f"🔄 Generating {num_questions} QA pairs for: {title}")
    
    qa_pairs = []
    used_question_types = set()
    
    # GUARANTEE a "complete_overview" question as first for maximum coverage
    overview_type = next((qt for qt in QUESTION_TYPES if qt["type"] == "complete_overview"), None)
    
    if overview_type:
        print(f"   🎯 Creating complete overview...")
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
                    print(f"   ✅ QA 1/{num_questions} created (complete_overview)")
                    break
            if attempt < 2:
                print(f"   ⚠️ Attempt {attempt + 1} failed, retrying...")
    
    # Prioritize certain question types based on content
    features = detect_content_features(lexikon_entry)
    priority_types = []
    
    if features['has_technical_data']:
        priority_types.append("technical_specs")
    if features['has_numbers']:
        priority_types.append("data_and_facts")
    if features['has_list'] or features['has_table']:
        priority_types.append("structure_and_components")
    
    # Add prioritized questions
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
                        print(f"   ✅ QA {len(qa_pairs)}/{num_questions} created ({priority_type})")
                        break
    
    # Fill with additional random question types
    remaining_questions = num_questions - len(qa_pairs)
    
    for i in range(remaining_questions):
        # Choose unused question types
        available_types = [qt for qt in QUESTION_TYPES if qt["type"] not in used_question_types]
        
        # If all were used, allow repetition
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
                    print(f"   ✅ QA {len(qa_pairs)}/{num_questions} created ({question_type['type']})")
                    break
        
        if len(qa_pairs) <= len(qa_pairs):
            print(f"   ⚠️ QA {len(qa_pairs)+1} could not be created")
    
    return qa_pairs

# ========================================
# GPT-OSS TAG FORMAT CONVERSION
# ========================================

def convert_to_gpt_oss_tag_format(qa_pair: Dict[str, Any], include_system: bool = True, include_analysis: bool = False, include_developer: bool = False) -> str:
    """
    Converts a single QA pair to GPT-OSS tag format.
    
    Format:
    <|start|>system<|message|>...<|end|>
    <|start|>developer<|message|>...<|end|>
    <|start|>user<|message|>...<|end|>
    <|start|>assistant<|channel|>analysis<|message|>...<|end|>
    <|start|>assistant<|channel|>final<|message|>...<|end|>
    """
    conversation_parts = []
    
    # Detect language for system prompt
    language = detect_language(qa_pair.get("answer", ""))
    
    # Add system message
    if include_system:
        system_type = random.choice(list(SYSTEM_PROMPTS.keys()))
        system_msg = SYSTEM_PROMPTS[system_type]
        
        # Add knowledge cutoff and reasoning instructions
        system_full = f"{system_msg}\nKnowledge cutoff: 2024-06\nCurrent date: 2025-08-22"
        
        if include_analysis:
            system_full += "\nReasoning: medium\n# Valid channels: analysis, commentary, final. Channel must be included for every message."
        
        conversation_parts.append(f"<|start|>system<|message|>{system_full}<|end|>")
    
    # Add developer message (optional)
    if include_developer and random.random() < 0.2:  # 20% chance for developer instructions
        if language == "German":
            developer_instructions = [
                "# Instructions\nreasoning language: German\nYou are an AI assistant with detailed technical knowledge.",
                "# Instructions\nreasoning language: German\nProvide comprehensive answers with all technical specifications.",
                "# Instructions\nreasoning language: German\nYou are a helpful assistant specializing in industrial products."
            ]
        else:
            developer_instructions = [
                "# Instructions\nreasoning language: English\nYou are an AI assistant with detailed technical knowledge.",
                "# Instructions\nreasoning language: English\nProvide comprehensive answers with all technical specifications.",
                "# Instructions\nreasoning language: English\nYou are a helpful assistant specializing in industrial products."
            ]
        
        developer_msg = random.choice(developer_instructions)
        conversation_parts.append(f"<|start|>developer<|message|>{developer_msg}<|end|>")
    
    # Add user message
    question = qa_pair["question"]
    conversation_parts.append(f"<|start|>user<|message|>{question}<|end|>")
    
    # Add assistant response with optional analysis channel
    if include_analysis and random.random() < 0.3:  # 30% chance to include analysis
        # Create analysis content based on question type
        analysis_content = generate_analysis_content(qa_pair)
        conversation_parts.append(f"<|start|>assistant<|channel|>analysis<|message|>{analysis_content}<|end|>")
    
    # Add final response
    answer = qa_pair["answer"]
    if include_analysis:
        conversation_parts.append(f"<|start|>assistant<|channel|>final<|message|>{answer}<|end|>")
    else:
        conversation_parts.append(f"<|start|>assistant<|message|>{answer}<|end|>")
    
    return "".join(conversation_parts)

def generate_analysis_content(qa_pair: Dict[str, Any]) -> str:
    """
    Generates analysis/reasoning content for the assistant's thought process.
    """
    title = qa_pair.get("title", "this topic")
    question_type = qa_pair.get("question_type", "general")
    language = detect_language(qa_pair.get("answer", ""))
    
    # Generate analysis based on question type and language
    if language == "German":
        analysis_templates = [
            f"Der Benutzer fragt nach {title}. Ich werde eine umfassende Antwort mit allen technischen Details geben.",
            f"Diese Frage bezieht sich auf {title}. Ich sollte alle relevanten Spezifikationen und Eigenschaften erläutern.",
            f"Ich verstehe, dass der Benutzer Informationen über {title} benötigt. Ich werde strukturiert antworten.",
            f"Die Anfrage betrifft {title}. Ich werde eine detaillierte Erklärung mit allen verfügbaren Daten liefern."
        ]
    else:
        analysis_templates = [
            f"The user is asking about {title}. I'll provide a comprehensive answer with all technical details.",
            f"This question relates to {title}. I should explain all relevant specifications and features.",
            f"I understand the user needs information about {title}. I'll structure my response clearly.",
            f"The query concerns {title}. I'll provide a detailed explanation with all available data."
        ]
    
    if question_type == "technical_specs":
        if language == "German":
            analysis_templates.append(f"Ich werde alle technischen Spezifikationen von {title} auflisten.")
        else:
            analysis_templates.append(f"I'll list all technical specifications for {title}.")
    elif question_type == "application":
        if language == "German":
            analysis_templates.append(f"Ich erkläre die praktischen Anwendungen von {title}.")
        else:
            analysis_templates.append(f"I'll explain the practical applications of {title}.")
    
    return random.choice(analysis_templates)

def create_multi_turn_gpt_oss_format(qa_group: List[Dict[str, Any]], max_turns: int = 3) -> str:
    """
    Creates a multi-turn conversation in GPT-OSS tag format.
    """
    conversation_parts = []
    num_turns = min(len(qa_group), max_turns)
    selected_qa = random.sample(qa_group, num_turns)
    
    # Detect language
    language = detect_language(selected_qa[0].get("answer", ""))
    
    # Add system message
    system_type = random.choice(list(SYSTEM_PROMPTS.keys()))
    system_msg = SYSTEM_PROMPTS[system_type]
    system_full = f"{system_msg}\nKnowledge cutoff: 2024-06\nCurrent date: 2025-08-22\nReasoning: enabled\n# Valid channels: analysis, commentary, final."
    conversation_parts.append(f"<|start|>system<|message|>{system_full}<|end|>")
    
    # Optionally add developer instructions
    if random.random() < 0.25:  # 25% chance for developer instructions in multi-turn
        if language == "German":
            developer_msg = f"# Instructions\nreasoning language: German\nYou are an AI assistant providing detailed technical information in a conversational manner."
        else:
            developer_msg = f"# Instructions\nreasoning language: English\nYou are an AI assistant providing detailed technical information in a conversational manner."
        conversation_parts.append(f"<|start|>developer<|message|>{developer_msg}<|end|>")
    
    # Add turns
    for i, qa in enumerate(selected_qa):
        # User question
        if i == 0:
            question = qa["question"]
        else:
            # Follow-up questions
            if language == "German":
                follow_ups = ["Können Sie auch erklären", "Was ist mit", "Erzählen Sie mir außerdem über", "Ich möchte auch wissen"]
            else:
                follow_ups = ["Can you also explain", "What about", "Additionally, tell me about", "I'd also like to know"]
            follow_up = random.choice(follow_ups)
            question = f"{follow_up} {qa['question'].lower()}"
        
        conversation_parts.append(f"<|start|>user<|message|>{question}<|end|>")
        
        # Assistant response with occasional analysis
        if random.random() < 0.2:  # 20% chance for analysis in multi-turn
            analysis = generate_analysis_content(qa)
            conversation_parts.append(f"<|start|>assistant<|channel|>analysis<|message|>{analysis}<|end|>")
            conversation_parts.append(f"<|start|>assistant<|channel|>final<|message|>{qa['answer']}<|end|>")
        else:
            conversation_parts.append(f"<|start|>assistant<|message|>{qa['answer']}<|end|>")
    
    return "".join(conversation_parts)

# ========================================
# HARMONY FORMAT CONVERSION
# ========================================

def convert_to_harmony_format(qa_pairs: List[Dict[str, Any]], include_system_prompt: bool = True) -> List[Dict[str, Any]]:
    """
    Converts QA pairs to Harmony format (GPT-OSS style).
    
    Harmony format structure:
    {
        "conversations": [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "answer"}
        ]
    }
    """
    harmony_dataset = []
    
    for qa in qa_pairs:
        # Create basic conversation
        conversation = []
        
        # Optionally add system message for context
        if include_system_prompt and random.random() < 0.3:  # Add system prompt to 30% of conversations
            system_type = random.choice(list(SYSTEM_PROMPTS.keys()))
            conversation.append({
                "from": "system",
                "value": SYSTEM_PROMPTS[system_type]
            })
        
        # Add human question - make it more conversational sometimes
        if random.random() < 0.4:  # 40% chance for conversational starter
            starter = random.choice(CONVERSATION_STARTERS)
            question = f"{starter} {qa['title'].lower()}. {qa['question']}"
        else:
            question = qa["question"]
        
        conversation.append({
            "from": "human",
            "value": question
        })
        
        # Add GPT response
        conversation.append({
            "from": "gpt",
            "value": qa["answer"]
        })
        
        # Create the harmony format entry
        harmony_entry = {
            "conversations": conversation,
            "metadata": {
                "source": qa.get("source", ""),
                "title": qa.get("title", ""),
                "question_type": qa.get("question_type", ""),
                "answer_length": len(qa["answer"]),
                "has_markdown": "**" in qa["answer"] or "*" in qa["answer"] or "#" in qa["answer"],
                "conversation_turns": len(conversation),
                "has_system_prompt": any(turn["from"] == "system" for turn in conversation)
            }
        }
        
        harmony_dataset.append(harmony_entry)
    
    return harmony_dataset

def create_multi_turn_conversations(qa_pairs: List[Dict[str, Any]], max_turns: int = 3) -> List[Dict[str, Any]]:
    """
    Creates multi-turn conversations from QA pairs for more natural dialogue flow.
    Groups related QA pairs into single conversations.
    """
    multi_turn_dataset = []
    
    # Group QA pairs by title
    grouped_qa = {}
    for qa in qa_pairs:
        title = qa.get("title", "Unknown")
        if title not in grouped_qa:
            grouped_qa[title] = []
        grouped_qa[title].append(qa)
    
    # Create multi-turn conversations for entries with multiple QA pairs
    for title, qa_group in grouped_qa.items():
        if len(qa_group) >= 2:
            # Create multi-turn conversation
            num_turns = min(len(qa_group), max_turns)
            selected_qa = random.sample(qa_group, num_turns)
            
            conversation = []
            
            # Optionally add system prompt
            if random.random() < 0.5:
                system_type = random.choice(list(SYSTEM_PROMPTS.keys()))
                conversation.append({
                    "from": "system",
                    "value": SYSTEM_PROMPTS[system_type]
                })
            
            # Add turns
            for i, qa in enumerate(selected_qa):
                # First turn can have conversation starter
                if i == 0 and random.random() < 0.6:
                    starter = random.choice(CONVERSATION_STARTERS)
                    question = f"{starter} {title.lower()}. {qa['question']}"
                else:
                    # Follow-up questions can be more casual
                    follow_ups = [
                        "Can you also explain",
                        "What about",
                        "Additionally, tell me about",
                        "I'd also like to know",
                        "Furthermore,"
                    ]
                    if i > 0 and random.random() < 0.5:
                        follow_up = random.choice(follow_ups)
                        question = f"{follow_up} {qa['question'].lower()}"
                    else:
                        question = qa["question"]
                
                conversation.append({
                    "from": "human",
                    "value": question
                })
                
                conversation.append({
                    "from": "gpt",
                    "value": qa["answer"]
                })
            
            # Create multi-turn entry
            multi_turn_entry = {
                "conversations": conversation,
                "metadata": {
                    "source": selected_qa[0].get("source", ""),
                    "title": title,
                    "question_types": [qa.get("question_type", "") for qa in selected_qa],
                    "total_answer_length": sum(len(qa["answer"]) for qa in selected_qa),
                    "conversation_turns": len(conversation),
                    "is_multi_turn": True,
                    "has_system_prompt": any(turn["from"] == "system" for turn in conversation)
                }
            }
            
            multi_turn_dataset.append(multi_turn_entry)
    
    return multi_turn_dataset

# ========================================
# MAIN PROCESSING
# ========================================

def process_lexikon_to_harmony_dataset(input_dir: str, output_dir: str, output_filename: str, generate_tag_format: bool = True):
    """Main function: Converts all lexicon files to a Harmony format dataset for GPT-OSS."""
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🚀 Starting conversion Lexicon → Harmony Format Dataset (GPT-OSS)")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📄 Output file (JSON): {output_filename}")
    if generate_tag_format:
        print(f"📄 Output file (Tagged): {OUTPUT_FILENAME_TAGGED}")
    print(f"🔧 API type: {api_name}")
    print(f"🔧 Server: {server_url}")
    print(f"🔧 Model: {model_name}")
    print(f"✨ Features:")
    print(f"   - Harmony format (GPT-OSS compatible): ✅")
    print(f"   - GPT-OSS Tag format: {'✅' if generate_tag_format else '❌'}")
    print(f"   - Multi-turn conversations: ✅")
    print(f"   - System prompts: ✅")
    print(f"   - Analysis channels: ✅")
    print(f"   - Conversational starters: ✅")
    print(f"   - Extended question types: {len(QUESTION_TYPES)} categories")
    print(f"   - Complete fact preservation: ✅")
    print(f"   - Markdown formatting: ✅")
    print(f"   - Language agnostic: ✅")
    
    # Check API connection
    if not check_api_connection():
        print("❌ API connection failed. Processing aborted.")
        return
    
    # Load lexicon entries
    lexikon_entries = load_lexikon_files(input_dir)
    
    if not lexikon_entries:
        print("❌ No lexicon entries found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating QA pairs for {len(lexikon_entries)} lexicon entries")
    print(f"{'='*60}")
    
    all_qa_pairs = []
    statistics = {
        'total_entries': len(lexikon_entries),
        'processed_entries': 0,
        'failed_entries': 0,
        'total_qa_pairs': 0,
        'qa_by_type': {}
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
                
                # Statistics by type
                for qa in qa_pairs:
                    q_type = qa.get('question_type', 'unknown')
                    statistics['qa_by_type'][q_type] = statistics['qa_by_type'].get(q_type, 0) + 1
                
                print(f"   📊 {len(qa_pairs)} QA pairs generated")
            else:
                statistics['failed_entries'] += 1
                print(f"   ⚠️ No QA pairs generated")
                
        except Exception as e:
            statistics['failed_entries'] += 1
            print(f"   ❌ Error during processing: {e}")
    
    # Convert to Harmony format
    print(f"\n🔄 Converting to Harmony format...")
    harmony_dataset = convert_to_harmony_format(all_qa_pairs, include_system_prompt=True)
    
    # Create multi-turn conversations
    print(f"🔄 Creating multi-turn conversations...")
    multi_turn_conversations = create_multi_turn_conversations(all_qa_pairs, max_turns=3)
    
    # Combine single-turn and multi-turn conversations
    all_conversations = harmony_dataset + multi_turn_conversations
    
    # Shuffle for better training distribution
    random.shuffle(all_conversations)
    
    # Generate GPT-OSS Tag Format if requested
    if generate_tag_format:
        print(f"\n🔄 Generating GPT-OSS Tag Format...")
        tagged_conversations = []
        
        # Group QA pairs by title for multi-turn
        grouped_qa = {}
        for qa in all_qa_pairs:
            title = qa.get("title", "Unknown")
            if title not in grouped_qa:
                grouped_qa[title] = []
            grouped_qa[title].append(qa)
        
        # Generate single-turn conversations in tag format
        for qa in all_qa_pairs:
            include_analysis = random.random() < 0.4  # 40% with analysis channel
            include_developer = random.random() < 0.3  # 30% with developer instructions
            tagged_conv = convert_to_gpt_oss_tag_format(
                qa, 
                include_system=True, 
                include_analysis=include_analysis,
                include_developer=include_developer
            )
            tagged_conversations.append(tagged_conv)
        
        # Generate multi-turn conversations in tag format
        for title, qa_group in grouped_qa.items():
            if len(qa_group) >= 2:
                multi_turn_tagged = create_multi_turn_gpt_oss_format(qa_group, max_turns=3)
                tagged_conversations.append(multi_turn_tagged)
        
        # Shuffle tagged conversations
        random.shuffle(tagged_conversations)
        
        # Save tagged format
        tagged_output_path = Path(output_dir) / OUTPUT_FILENAME_TAGGED
        with open(tagged_output_path, 'w', encoding='utf-8') as f:
            for conv in tagged_conversations:
                f.write(conv)
                f.write("\n\n")  # Separate conversations with double newline
        
        print(f"✅ Tagged format saved: {tagged_output_path}")
        print(f"   - Total conversations: {len(tagged_conversations)}")
    
    # Calculate additional statistics
    total_conversations = len(all_conversations)
    multi_turn_count = sum(1 for conv in all_conversations if conv['metadata'].get('is_multi_turn', False))
    system_prompt_count = sum(1 for conv in all_conversations if conv['metadata'].get('has_system_prompt', False))
    avg_turns = sum(conv['metadata']['conversation_turns'] for conv in all_conversations) / total_conversations if total_conversations > 0 else 0
    avg_answer_length = sum(
        conv['metadata'].get('answer_length', conv['metadata'].get('total_answer_length', 0)) 
        for conv in all_conversations
    ) / total_conversations if total_conversations > 0 else 0
    markdown_count = sum(1 for conv in all_conversations if conv['metadata'].get('has_markdown', False))
    
    # Create final dataset in Harmony format
    output_path = Path(output_dir) / output_filename
    
    final_dataset = {
        "metadata": {
            "format": "harmony_gpt_oss",
            "version": "1.0",
            "total_lexicon_entries": statistics['total_entries'],
            "processed_entries": statistics['processed_entries'],
            "failed_entries": statistics['failed_entries'],
            "total_qa_pairs_generated": statistics['total_qa_pairs'],
            "total_conversations": total_conversations,
            "single_turn_conversations": total_conversations - multi_turn_count,
            "multi_turn_conversations": multi_turn_count,
            "conversations_with_system_prompt": system_prompt_count,
            "avg_conversation_turns": avg_turns,
            "avg_answer_length": avg_answer_length,
            "markdown_formatted_answers": markdown_count,
            "qa_by_type": statistics['qa_by_type'],
            "api_used": api_name,
            "model_used": model_name,
            "server_used": server_url,
            "tagged_format_generated": generate_tag_format,
            "features": {
                "harmony_format": True,
                "gpt_oss_compatible": True,
                "gpt_oss_tag_format": generate_tag_format,
                "multi_turn_support": True,
                "system_prompts": True,
                "analysis_channels": True,
                "conversational_starters": True,
                "markdown_formatting": True,
                "complete_overview_guaranteed": True,
                "fact_preservation": True,
                "table_support": True,
                "language_agnostic": True,
                "extended_question_types": len(QUESTION_TYPES)
            }
        },
        "conversations": all_conversations
    }
    
    # Save dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    
    # Output statistics
    print(f"\n{'='*60}")
    print(f"🎉 Harmony Format Dataset (GPT-OSS) successfully created!")
    print(f"{'='*60}")
    print(f"📊 Detailed Statistics:")
    print(f"   📁 Processing:")
    print(f"      - Total lexicon entries: {statistics['total_entries']}")
    print(f"      - Successfully processed: {statistics['processed_entries']}")
    print(f"      - Failed: {statistics['failed_entries']}")
    print(f"   💬 Conversations:")
    print(f"      - Total conversations: {total_conversations}")
    print(f"      - Single-turn: {total_conversations - multi_turn_count}")
    print(f"      - Multi-turn: {multi_turn_count}")
    print(f"      - With system prompts: {system_prompt_count}")
    print(f"      - Average turns per conversation: {avg_turns:.1f}")
    print(f"   📝 Content:")
    print(f"      - Total QA pairs generated: {statistics['total_qa_pairs']}")
    print(f"      - Average answer length: {avg_answer_length:.0f} characters")
    print(f"      - With Markdown formatting: {markdown_count}/{total_conversations}")
    print(f"   🎯 Question types distribution:")
    for q_type, count in sorted(statistics['qa_by_type'].items(), key=lambda x: x[1], reverse=True):
        print(f"      - {q_type}: {count}")
    print(f"   🔧 Technical details:")
    print(f"      - Format: Harmony (GPT-OSS compatible)")
    print(f"      - Tagged format: {'Generated' if generate_tag_format else 'Not generated'}")
    print(f"      - API: {api_name}")
    print(f"      - Server: {server_url}")
    print(f"      - Model: {model_name}")
    print(f"   💾 Output:")
    print(f"      - JSON file: {output_path}")
    print(f"      - JSON size: {output_path.stat().st_size / 1024:.1f} KB" if output_path.exists() else "")
    if generate_tag_format:
        tagged_path = Path(output_dir) / OUTPUT_FILENAME_TAGGED
        print(f"      - Tagged file: {tagged_path}")
        print(f"      - Tagged size: {tagged_path.stat().st_size / 1024:.1f} KB" if tagged_path.exists() else "")
    
    # Show examples
    if all_conversations:
        print(f"\n📋 Example conversations (JSON format):")
        
        # Show a single-turn example
        single_turn_examples = [conv for conv in all_conversations if not conv['metadata'].get('is_multi_turn', False)]
        if single_turn_examples:
            example = single_turn_examples[0]
            print(f"\n   🔹 Single-turn example:")
            for turn in example['conversations'][:3]:  # Show first 3 turns
                role = turn['from']
                value_preview = turn['value'][:150] + "..." if len(turn['value']) > 150 else turn['value']
                print(f"      {role}: {value_preview}")
    
    if generate_tag_format and 'tagged_conversations' in locals():
        print(f"\n📋 Example conversation (Tag format):")
        if tagged_conversations:
            example_tagged = tagged_conversations[0]
            # Show first 800 characters of tagged format, properly formatted
            lines = example_tagged.split('<|end|>')
            for i, line in enumerate(lines[:5]):  # Show first 5 lines
                if line:
                    formatted_line = line.replace('<|start|>', '\n   <|start|>')
                    print(f"   {formatted_line}<|end|>")
            if len(lines) > 5:
                print(f"   ... (conversation continues)")

# ========================================
# MAIN PROGRAM
# ========================================

if __name__ == "__main__":
    # Import regex for extended pattern matching
    import re
    
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🎯 HARMONY FORMAT CONVERTER FOR GPT-OSS")
    print(f"{'='*60}")
    print(f"🔧 CONFIGURATION (from central config):")
    print(f"   - API type: {api_name}")
    print(f"   - Server: {server_url}")
    print(f"   - Model: {model_name}")
    print(f"   - Output format: Harmony (GPT-OSS compatible)")
    print(f"   - Tag format: Enabled (GPT-OSS native)")
    print(f"   - Multi-turn conversations: ✅")
    print(f"   - System prompts: ✅")
    print(f"   - Analysis channels: ✅")
    print(f"   - Extended features: ✅")
    print(f"   - Fact preservation: ✅")
    print(f"   - Markdown formatting: ✅")
    print(f"   - Language agnostic: ✅")
    
    # Start processing with tag format generation enabled
    process_lexikon_to_harmony_dataset(INPUT_DIR, OUTPUT_DIR, OUTPUT_FILENAME, generate_tag_format=True)