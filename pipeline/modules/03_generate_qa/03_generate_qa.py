import json
import requests
import random
import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any

# ========================================
# LOAD CENTRAL CONFIGURATION
# ========================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))  # To main directory
from config_loader import PipelineConfigLoader

# Load configuration for this module
config_loader = PipelineConfigLoader()
module_config = config_loader.get_module_config("03_generate_qa")
pipeline_config = config_loader.get_pipeline_config()

# Extract configuration values
API_BASE_URL = module_config.get("api_base_url", "")
API_KEY = module_config.get("api_key", "")
MODEL_NAME = module_config.get("model_name", "")

# Directories
INPUT_DIR = "INPUT"
OUTPUT_DIR = "OUTPUT"
OUTPUT_FILENAME = "dataset.jsonl"

# Show loaded configuration
print("=" * 60)
print("📋 CONFIGURATION LOADED (03_generate_qa)")
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
        print(f"💡 Check: Server running on {API_BASE_URL}?")
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
Create a high-quality question-answer pair for an instruction dataset.

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
    """Sends a request to the LLM API and retrieves the response."""
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system", 
                "content": "You are an expert at creating high-quality question-answer pairs for instruction datasets. Your answers are complete, fact-rich and preserve ALL details from the source. You use Markdown formatting for better structure. You respond in the same language as the source document."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,  # Increased for longer, complete answers
        "temperature": 0.3   # Low for factual accuracy
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
            print(f"❌ API error (attempt {attempt + 1}): {e}")
    
    return None

# ========================================
# QA GENERATION
# ========================================

def extract_qa_from_response(response: str) -> Optional[Dict[str, str]]:
    """Extracts QA pair from the API response."""
    try:
        import re
        # Clean the response
        response = response.strip()
        
        qa_data = None
        
        # Method 1: Try finding a markdown JSON block
        json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response)
        if json_match:
            try:
                qa_data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
                
        # Method 2: Try finding any JSON object containing "question" and "answer"
        if not qa_data:
            json_match = re.search(r'({[\s\S]*?"question"[\s\S]*?"answer"[\s\S]*?})', response)
            if json_match:
                try:
                    qa_data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # Method 3: Direct parse as last resort
        if not qa_data:
            try:
                # Remove code block markers if present manually
                clean_resp = response
                if clean_resp.startswith("```json") and clean_resp.endswith("```"):
                    clean_resp = clean_resp[7:-3].strip()
                elif clean_resp.startswith("```") and clean_resp.endswith("```"):
                    clean_resp = clean_resp[3:-3].strip()
                qa_data = json.loads(clean_resp)
            except json.JSONDecodeError:
                pass

        if not qa_data:
            print(f"❌ Invalid QA format: Could not extract valid JSON")
            return None
        
        # Validate structure
        if "question" in qa_data and "answer" in qa_data:
            return {
                "question": str(qa_data["question"]).strip(),
                "answer": str(qa_data["answer"]).strip()
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
            num_questions = random.randint(2, 3)
        elif content_length < 1500:
            num_questions = random.randint(3, 4)
        else:
            num_questions = random.randint(4, 5)
    
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
# BIDIRECTIONAL / INVERSE QA GENERATION
# ========================================

def generate_inverse_qa_prompt(original_question: str, original_answer: str) -> str:
    """Creates a prompt to generate inverse/bidirectional QA pairs from an existing QA pair.
    
    This is the core innovation of OpenTuneWeaver: if the model learns that
    'Tom Cruise's mother is Jay Dee', it should also learn that
    'Jay Dee's son is Tom Cruise'. This ensures bidirectional knowledge.
    """
    prompt = f"""You are an expert at creating inverse question-answer pairs for training datasets.

Given the following question-answer pair, your task is to CREATE NEW QUESTIONS that approach
the same knowledge FROM THE OPPOSITE DIRECTION.

The goal: If the original teaches "A relates to B", the inverse should teach "B relates to A".

Examples of this principle:
- Original: "What is the capital of France?" → "Paris"
  Inverse: "Which country has Paris as its capital?" → "France"
- Original: "Who invented the telephone?" → "Alexander Graham Bell"
  Inverse: "What did Alexander Graham Bell invent?" → "The telephone"
- Original: "What temperature does water boil at?" → "100°C at standard pressure"
  Inverse: "What happens to water at 100°C?" → "It reaches its boiling point at standard pressure"

Now apply this principle:

ORIGINAL QUESTION:
{original_question}

ORIGINAL ANSWER:
{original_answer}

INSTRUCTIONS:
1. Identify the KEY ENTITIES, FACTS, NUMBERS, or CONCEPTS mentioned in the answer
2. For each significant entity/fact, create a NEW question where that entity becomes the SUBJECT
3. The new answer should reference the original question's subject
4. Respond in the SAME LANGUAGE as the original Q&A
5. Generate 1 to 3 inverse pairs (depending on how many reversible facts the answer contains)
6. Each answer should be complete and self-contained (3+ sentences)

OUTPUT FORMAT (JSON array):
[
  {{
    "question": "The reversed/inverse question",
    "answer": "The complete answer from the reversed perspective"
  }}
]

IMPORTANT: Respond ONLY with the JSON array, no additional explanations!
If the Q&A pair does not contain reversible relationships (e.g., purely abstract concepts),
return an empty array: []
"""
    return prompt


def generate_inverse_qa(qa_pairs: List[Dict[str, Any]], max_inverse_per_pair: int = 2) -> List[Dict[str, Any]]:
    """Generates inverse/bidirectional QA pairs from existing QA pairs.
    
    This is OpenTuneWeaver's core innovation: ensuring the fine-tuned model
    learns knowledge bidirectionally. For every fact A→B, we also teach B→A.
    """
    if not qa_pairs:
        return []
    
    inverse_pairs = []
    
    for idx, qa in enumerate(qa_pairs):
        question = qa.get('question', '')
        answer = qa.get('answer', '')
        
        if not question or not answer:
            continue
        
        # Skip very short answers (not enough content to reverse)
        if len(answer) < 50:
            continue
        
        print(f"   🔄 Generating inverse QA for pair {idx + 1}...")
        
        prompt = generate_inverse_qa_prompt(question, answer)
        response = submit_to_api(prompt)
        
        if not response:
            continue
        
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            elif response.startswith("```") and response.endswith("```"):
                response = response[3:-3].strip()
            
            inverse_data = json.loads(response)
            
            if not isinstance(inverse_data, list):
                continue
            
            # Limit inverse pairs per original
            for inv_idx, inv_pair in enumerate(inverse_data[:max_inverse_per_pair]):
                if 'question' in inv_pair and 'answer' in inv_pair:
                    inv_qa = {
                        'question': inv_pair['question'].strip(),
                        'answer': inv_pair['answer'].strip(),
                        'source': qa.get('source', ''),
                        'title': qa.get('title', ''),
                        'question_type': 'inverse_bidirectional',
                        'original_question': question[:100]  # Reference to original
                    }
                    inverse_pairs.append(inv_qa)
                    print(f"      ✅ Inverse QA {inv_idx + 1}: {inv_qa['question'][:60]}...")
        
        except json.JSONDecodeError:
            print(f"      ⚠️ Could not parse inverse QA response")
            continue
        except Exception as e:
            print(f"      ⚠️ Error generating inverse QA: {e}")
            continue
    
    return inverse_pairs


# ========================================
# DATASET CONVERSION
# ========================================

def convert_to_chat_masterformat(qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts QA pairs to the generic Chat-Masterformat (JSONL)."""
    dataset = []
    
    for qa in qa_pairs:
        # Create standard system prompt if no specific one exists
        system_content = "You are a helpful and knowledgeable assistant."
        
        chat_entry = {
            "messages": [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": qa["question"]
                },
                {
                    "role": "assistant",
                    "content": qa["answer"]
                }
            ],
            "metadata": {
                "source": qa.get("source", ""),
                "title": qa.get("title", ""),
                "question_type": qa.get("question_type", ""),
            }
        }
        dataset.append(chat_entry)
    
    return dataset

# ========================================
# MAIN PROCESSING
# ========================================

def process_lexikon_to_qa_dataset(input_dir: str, output_dir: str, output_filename: str):
    """Main function: Converts all lexicon files to a comprehensive Chat-Masterformat dataset."""
    print(f"🚀 Starting extended conversion Lexicon → Chat-Masterformat Dataset")
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📄 Output file: {output_filename}")
    print(f"🔧 API type: LLM API")
    print(f"🔧 Server: {API_BASE_URL}")
    print(f"🔧 Model: {MODEL_NAME}")
    print(f"✨ Features:")
    print(f"   - Extended question types: {len(QUESTION_TYPES)} categories")
    print(f"   - Bidirectional/Inverse QA: ✅ (Core Innovation)")
    print(f"   - Complete fact preservation: ✅")
    print(f"   - Markdown formatting: ✅")
    print(f"   - Dynamic question count: ✅")
    print(f"   - Table support: ✅")
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
    
    total_entries: int = len(lexikon_entries)
    processed_entries: int = 0
    failed_entries: int = 0
    total_qa_pairs_count: int = 0
    total_inverse_count: int = 0
    qa_by_type: Dict[str, int] = {}
    
    # Process each entry
    for idx, entry in enumerate(lexikon_entries, 1):
        title = entry.get('title', entry.get('name', f'Entry {idx}'))
        print(f"\n🔄 Entry {idx}/{len(lexikon_entries)}: {title}")
        
        try:
            qa_pairs = generate_qa_for_entry(entry)
            
            if qa_pairs:
                all_qa_pairs.extend(qa_pairs)
                processed_entries += 1
                total_qa_pairs_count += len(qa_pairs)
                
                # Statistics by type
                for qa in qa_pairs:
                    q_type = qa.get('question_type', 'unknown')
                    qa_by_type[q_type] = qa_by_type.get(q_type, 0) + 1
                
                print(f"   📊 {len(qa_pairs)} QA pairs generated")
                
                # === BIDIRECTIONAL QA (Core Innovation) ===
                print(f"   🔄 Generating bidirectional/inverse QA pairs...")
                inverse_pairs = generate_inverse_qa(qa_pairs)
                if inverse_pairs:
                    all_qa_pairs.extend(inverse_pairs)
                    total_inverse_count += len(inverse_pairs)
                    total_qa_pairs_count += len(inverse_pairs)
                    for inv in inverse_pairs:
                        q_type = inv.get('question_type', 'inverse_bidirectional')
                        qa_by_type[q_type] = qa_by_type.get(q_type, 0) + 1
                    print(f"   🔁 {len(inverse_pairs)} inverse QA pairs added (bidirectional knowledge)")
                else:
                    print(f"   ℹ️ No inverse QA pairs generated for this entry")
            else:
                failed_entries += 1
                print(f"   ⚠️ No QA pairs generated")
                
        except Exception as e:
            failed_entries += 1
            print(f"   ❌ Error during processing: {e}")
    
    # Convert to Chat-Masterformat
    chat_dataset = convert_to_chat_masterformat(all_qa_pairs)
    
    # Calculate additional statistics
    avg_answer_length = 0
    markdown_count = 0
    if chat_dataset:
        total_len = sum(len(entry['messages'][2]['content']) for entry in chat_dataset if len(entry.get('messages', [])) > 2)
        avg_answer_length = total_len / len(chat_dataset)
        markdown_count = sum(1 for entry in chat_dataset if len(entry.get('messages', [])) > 2 and ("**" in entry['messages'][2]['content'] or "*" in entry['messages'][2]['content'] or "#" in entry['messages'][2]['content']))
    
    # Create final dataset file
    output_path = Path(output_dir) / output_filename
    
    # Save dataset as JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in chat_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Output statistics
    print(f"\n{'='*60}")
    print(f"🎉 Chat-Masterformat Dataset successfully created!")
    print(f"{'='*60}")
    print(f"📊 Detailed Statistics:")
    print(f"   📁 Processing:")
    print(f"      - Total lexicon entries: {total_entries}")
    print(f"      - Successfully processed: {processed_entries}")
    print(f"      - Failed: {failed_entries}")
    print(f"   📝 QA pairs:")
    print(f"      - Total generated: {total_qa_pairs_count}")
    print(f"      - Thereof inverse/bidirectional: {total_inverse_count}")
    print(f"      - Average per entry: {total_qa_pairs_count/processed_entries:.1f}" if processed_entries > 0 else "")
    print(f"      - Average answer length: {avg_answer_length:.0f} characters")
    print(f"      - With Markdown formatting: {markdown_count}/{len(chat_dataset)}")
    print(f"   🎯 Question types:")
    for q_type, count in sorted(qa_by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"      - {q_type}: {count}")
    print(f"   🔧 Technical details:")
    print(f"      - API: {API_BASE_URL}")
    print(f"      - Model: {MODEL_NAME}")
    print(f"   💾 Output:")
    print(f"      - File: {output_path}")
    print(f"      - Size: {output_path.stat().st_size / 1024:.1f} KB" if output_path.exists() else "")
    
    # Show examples
    if chat_dataset:
        print(f"\n📋 Example generic chats:")
        
        examples_shown = 0
        
        for entry in chat_dataset:
            if examples_shown < 1:
                print(f"\n   User: {entry['messages'][1]['content']}")
                answer_preview = entry['messages'][2]['content'][:200] + "..." if len(entry['messages'][2]['content']) > 200 else entry['messages'][2]['content']
                print(f"   Assistant: {answer_preview}")
                examples_shown += 1

# ========================================
# MAIN PROGRAM
# ========================================

if __name__ == "__main__":
    # Import regex for extended pattern matching
    import re
    
    print(f"🔧 CONFIGURATION (from central config):")
    print(f"   - API type: LLM API")
    print(f"   - Server: {API_BASE_URL}")
    print(f"   - Model: {MODEL_NAME}")
    print(f"   - Extended features: ✅")
    print(f"   - Fact preservation: ✅")
    print(f"   - Markdown formatting: ✅")
    print(f"   - Table support: ✅")
    print(f"   - Language agnostic: ✅")
    
    # Start processing
    process_lexikon_to_qa_dataset(INPUT_DIR, OUTPUT_DIR, OUTPUT_FILENAME)