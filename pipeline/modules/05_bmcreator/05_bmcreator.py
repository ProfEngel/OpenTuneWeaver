import json
import requests
import random
import os
import sys
from pathlib import Path

# ========================================
# LOAD CENTRAL CONFIGURATION
# ========================================
sys.path.append(str(Path(__file__).parent.parent.parent))  # To main directory
from config_loader import PipelineConfigLoader

# Load configuration for this module
config_loader = PipelineConfigLoader("05_bmcreator")
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

# Directories
INPUT_DIR = "INPUT"
OUTPUT_DIR = "BENCHMARKFRAGEN"
OUTPUT_FILENAME = "benchmark_fragen_complete.json"

# Category mapping based on lexicon files
CATEGORY_MAPPING = {
    "CON01_Jahresabschluss": "Jahresabschlussanalyse",
    "CON02_Liquiditatsplanung": "Liquiditätsplanung", 
    "CON03_Budgetierung": "Budgetierung"
}

# Dynamic question count configuration from config
MAX_TOTAL_QUESTIONS = config.get("max_total_questions", 100)
MIN_QUESTIONS_PER_CATEGORY = config.get("min_questions_per_category", 5)
MAX_QUESTIONS_PER_CATEGORY = config.get("max_questions_per_category", 10)

# Question type configuration from config
QUESTION_TYPE_DISTRIBUTION = config.get("question_type_distribution", {
    "definition": 0.7,
    "transfer": 0.3
})

# Show loaded configuration
print("=" * 60)
print("📋 CONFIGURATION LOADED (05_bmcreator)")
print("=" * 60)
config_loader.print_config_summary()
print(f"  📊 Max total questions: {MAX_TOTAL_QUESTIONS}")
print(f"  📊 Min questions per category: {MIN_QUESTIONS_PER_CATEGORY}")
print(f"  📊 Max questions per category: {MAX_QUESTIONS_PER_CATEGORY}")
print(f"  📊 Distribution: {int(QUESTION_TYPE_DISTRIBUTION['definition']*100)}% Definition, {int(QUESTION_TYPE_DISTRIBUTION['transfer']*100)}% Transfer")
print("=" * 60)

# Question templates for different types (examples for reference)
DEFINITION_QUESTION_EXAMPLES = [
    "What is meant by {title}?",
    "How is {title} defined?",
    "Explain the concept of {title}.",
    "What is {title}?",
    "Define {title}.",
    "How is {title} determined?",
    "What are the main characteristics of {title}?",
    "Explain {title}.",
    "Describe {title}.",
    "What significance does {title} have?",
    "What characterizes {title}?",
    "How does {title} work?",
    "What does {title} consist of?",
    "What does {title} encompass?",
    "How is {title} structured?"
]

TRANSFER_QUESTION_EXAMPLES = [
    "A company A wants to implement {title}. How should it proceed?",
    "What steps should be considered when applying {title} in a medium-sized company?",
    "A controller must evaluate {title} for his company. What should he pay attention to?",
    "How can a company implement {title} in practice?",
    "A CFO asks you about {title}. How do you explain the practical relevance to him?",
    "What challenges can arise when implementing {title}?",
    "A startup wants to introduce {title}. What recommendations do you give?",
    "How does {title} affect business practice?",
    "A corporation plans to optimize {title}. Which factors are decisive?",
    "What practical effects does {title} have on controlling?",
    "An SME has problems with {title}. How can these be solved?",
    "Why is {title} important for companies and how is it applied?"
]

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

def load_lexikon_files_by_category(input_dir):
    """Loads all lexicon JSON files and groups them by categories."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory '{input_dir}' does not exist!")
        return {}
    
    lexikon_files = list(input_path.glob("lexikon_*.json"))
    print(f"📁 Found lexicon files: {len(lexikon_files)}")
    
    categories = {}
    
    for file in lexikon_files:
        print(f"📖 Loading: {file.name}")
        
        # Determine category from filename
        category = "Sonstiges"  # Fallback
        for key, value in CATEGORY_MAPPING.items():
            if key in file.name:
                category = value
                break
        
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entries = data.get('lexikon_entries', [])
                print(f"   - {len(entries)} entries found → Category: {category}")
                
                if category not in categories:
                    categories[category] = []
                categories[category].extend(entries)
                
        except Exception as e:
            print(f"❌ Error loading {file.name}: {e}")
    
    # Show category overview
    print(f"\n📊 Category overview:")
    for category, entries in categories.items():
        print(f"   - {category}: {len(entries)} entries")
    
    return categories

def select_question_type():
    """Selects question type based on probability distribution."""
    rand = random.random()
    if rand < QUESTION_TYPE_DISTRIBUTION["definition"]:
        return "definition"
    else:
        return "transfer"

def generate_benchmark_question_prompt(title, lexikon_entry, category, question_type=None):
    """Generates a prompt for creating a benchmark question."""
    
    # Choose question type if not specified
    if question_type is None:
        question_type = select_question_type()
    
    # Detect language of the source content
    source_language = detect_language(lexikon_entry)
    
    if question_type == "definition":
        instruction = f"""
Create a precise definition question and answer it professionally correct.

IMPORTANT - Answer Requirements:
- Use EXCLUSIVELY information from the provided lexicon entry
- Add NO external knowledge or personal interpretation
- Use ALL relevant information from the lexicon entry
- The answer must be comprehensive and complete
- Base your answer 100% on the given lexicon content

Task:
1. Create an appropriate definition question for the topic
2. Answer the question structured and completely based ONLY on the lexicon entry
3. The answer should cover all important aspects from the lexicon entry
4. Focus on: Definition, characteristics, functionality, structure (everything from the lexicon entry)
5. Style: Professional, precise, factual
6. Answer length: As long as necessary to cover all relevant information from the lexicon entry
7. CRITICAL: Respond in the same language as the source document. The source appears to be in {source_language}, so respond in {source_language}."""
    
    else:  # transfer
        instruction = f"""
Create a practice-oriented transfer question and answer it application-related.

IMPORTANT - Answer Requirements:
- Use EXCLUSIVELY information from the provided lexicon entry
- Add NO external knowledge or personal interpretation
- Use ALL relevant information from the lexicon entry
- The answer must be comprehensive and complete
- Base your answer 100% on the given lexicon content

Task:
1. Create an appropriate practical application question for the topic
2. Answer the question with practical reference based ONLY on the lexicon entry
3. The answer should cover all application-relevant aspects from the lexicon entry
4. Focus on: Practical significance/application (everything from the lexicon entry)
5. Style: Practice-oriented, advisory, implementation-focused
6. Answer length: As long as necessary to cover all relevant information from the lexicon entry
7. CRITICAL: Respond in the same language as the source document. The source appears to be in {source_language}, so respond in {source_language}."""

    prompt = f"""
You are an expert in {category} and should create a benchmark question.

CRITICALLY IMPORTANT - Use as answer source EXCLUSIVELY the following lexicon entry:
==========================================
Title: {title}
Content: {lexikon_entry}
==========================================

LANGUAGE DETECTION: The source content appears to be in {source_language}.

STRICTLY FOLLOW:
- Use for the answer ONLY information from the above lexicon entry
- Add NO external knowledge, no personal interpretations or assumptions
- Use ALL relevant information from the lexicon entry
- The answer must be based completely on the lexicon entry
- Leave out NO important aspects from the lexicon entry

{instruction}

Category context: {category}
Question type: {question_type}

CRITICAL: Respond in the same language as the source document. If the source is in German, respond in German. If the source is in English, respond in English.

Respond in the following JSON format:
{{
    "frage": "Your benchmark question here (appropriate for the topic and type)",
    "antwort": "Your comprehensive answer here - based ONLY on the lexicon entry"
}}

Respond ONLY with the JSON, without additional explanations.
"""
    return prompt

def submit_to_api(prompt, retries=3):
    """Sends a request to the chosen API and retrieves the response."""
    if USE_OPENAI_API:
        return submit_to_openai_api(prompt, retries)
    else:
        return submit_to_ollama_api(prompt, retries)

def submit_to_openai_api(prompt, retries=3):
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
                "content": "You are an expert at creating benchmark questions in financial controlling. Create precise, professionally demanding questions with complete answers. CRITICALLY IMPORTANT: Use for all answers EXCLUSIVELY the provided lexicon entries as knowledge source. NEVER add external knowledge. Respond in the same language as the source document."
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
                print(f"❌ OpenAI API error {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            print(f"❌ API error (attempt {attempt + 1}): {e}")
    
    return None

def submit_to_ollama_api(prompt, retries=3):
    """Sends a request to the Ollama API and retrieves the response."""
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
                "content": "You are an expert at creating benchmark questions in financial controlling. Create precise, professionally demanding questions with complete answers. CRITICALLY IMPORTANT: Use for all answers EXCLUSIVELY the provided lexicon entries as knowledge source. NEVER add external knowledge. Respond in the same language as the source document."
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
                print(f"❌ Ollama API error {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            print(f"❌ API error (attempt {attempt + 1}): {e}")
    
    return None

def extract_benchmark_qa_from_response(response):
    """Extracts benchmark question-answer pair from the API response."""
    try:
        # Clean the response
        response = response.strip()
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        
        qa_data = json.loads(response)
        
        # Validate the structure
        if "frage" in qa_data and "antwort" in qa_data:
            return {
                "frage": qa_data["frage"].strip(),
                "antwort": qa_data["antwort"].strip()
            }
        else:
            print(f"❌ Invalid benchmark QA format: {list(qa_data.keys())}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print(f"Raw Response (first 200 chars): {response[:200]}...")
        return None

def calculate_questions_per_category(categories_data):
    """Dynamically calculates the number of questions per category based on available entries."""
    total_entries = sum(len(entries) for entries in categories_data.values())
    num_categories = len(categories_data)
    
    print(f"📊 Dynamic question distribution:")
    print(f"   - Total categories: {num_categories}")
    print(f"   - Total entries: {total_entries}")
    print(f"   - Maximum questions: {MAX_TOTAL_QUESTIONS}")
    
    questions_distribution = {}
    
    # Calculate questions per category based on entries
    for category, entries in categories_data.items():
        num_entries = len(entries)
        
        if num_entries <= 10:
            # Small categories: 5-10 questions
            num_questions = min(num_entries, random.randint(MIN_QUESTIONS_PER_CATEGORY, MAX_QUESTIONS_PER_CATEGORY))
        elif num_entries <= 30:
            # Medium categories: 10-20 questions
            num_questions = min(num_entries, random.randint(10, 20))
        else:
            # Large categories: 15-35 questions
            num_questions = min(num_entries, random.randint(15, 35))
        
        questions_distribution[category] = num_questions
        print(f"   - {category}: {num_questions} questions (from {num_entries} entries)")
    
    # Check total and adjust if necessary
    total_planned = sum(questions_distribution.values())
    
    if total_planned > MAX_TOTAL_QUESTIONS:
        print(f"   ⚠️ Planned: {total_planned} questions → Reducing to {MAX_TOTAL_QUESTIONS}")
        
        # Proportional reduction
        reduction_factor = MAX_TOTAL_QUESTIONS / total_planned
        
        for category in questions_distribution:
            original = questions_distribution[category]
            questions_distribution[category] = max(1, int(original * reduction_factor))
        
        # Fine-tuning if questions remain
        current_total = sum(questions_distribution.values())
        remaining = MAX_TOTAL_QUESTIONS - current_total
        
        categories_list = list(questions_distribution.keys())
        for i in range(remaining):
            category = categories_list[i % len(categories_list)]
            if questions_distribution[category] < len(categories_data[category]):
                questions_distribution[category] += 1
    
    final_total = sum(questions_distribution.values())
    print(f"   ✅ Final: {final_total} questions distributed")
    
    return questions_distribution

def generate_benchmark_questions_for_category(category, entries, num_questions):
    """Generates benchmark questions for a category."""
    print(f"\n🔄 Generating {num_questions} benchmark questions for category: {category}")
    print(f"   📊 Available entries: {len(entries)}")
    
    # Calculate number of definition vs. transfer questions
    num_definition = int(num_questions * QUESTION_TYPE_DISTRIBUTION["definition"])
    num_transfer = num_questions - num_definition
    
    print(f"   🎯 Question type distribution: {num_definition} definitions, {num_transfer} transfer")
    
    if len(entries) < num_questions:
        print(f"   ⚠️ Only {len(entries)} entries available, generating {len(entries)} questions")
        num_questions = len(entries)
    
    # Random selection of entries (IMPORTANT: Not the first ones!)
    selected_entries = random.sample(entries, num_questions)
    print(f"   🎲 {num_questions} entries randomly selected")
    
    # Create list of question types
    question_types = (["definition"] * num_definition + ["transfer"] * num_transfer)
    random.shuffle(question_types)  # Shuffle the order
    
    benchmark_questions = []
    definition_count = 0
    transfer_count = 0
    
    for idx, entry in enumerate(selected_entries, 1):
        title = entry.get('title', f'Entry {idx}')
        lexikon_entry = entry.get('lexikon_entry', '')
        
        if not title or not lexikon_entry:
            print(f"   ⚠️ Incomplete entry skipped: {title}")
            continue
        
        # Determine question type for this question
        question_type = question_types[idx-1] if idx-1 < len(question_types) else select_question_type()
        
        print(f"   🔄 Question {idx}/{num_questions}: {title} ({question_type})")
        
        # Generate benchmark question
        prompt = generate_benchmark_question_prompt(title, lexikon_entry, category, question_type)
        
        for attempt in range(3):
            response = submit_to_api(prompt)
            if response:
                qa_pair = extract_benchmark_qa_from_response(response)
                if qa_pair:
                    # Determine ID based on category
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
                    
                    # Count question types
                    if question_type == "definition":
                        definition_count += 1
                    else:
                        transfer_count += 1
                    
                    print(f"      ✅ {question_type.title()} question created")
                    break
            print(f"      ❌ Attempt {attempt + 1} failed")
        
        if len(benchmark_questions) < idx:
            print(f"      ⚠️ Benchmark question {idx} could not be created")
    
    print(f"   📊 {len(benchmark_questions)} benchmark questions successfully generated")
    print(f"   📋 Actual distribution: {definition_count} definitions, {transfer_count} transfer")
    return benchmark_questions

def create_benchmark_dataset(categories_data):
    """Creates the complete benchmark dataset in the desired format."""
    
    dataset = {
        "titel": "Benchmark-Fragen Finanzcontrolling",
        "beschreibung": "Sammlung von Benchmark-Fragen für das Finanzcontrolling in den Bereichen Jahresabschlussanalyse, Budgetierung und Liquiditätsplanung",
        "kategorien": []
    }
    
    # Calculate dynamic question distribution
    questions_distribution = calculate_questions_per_category(categories_data)
    
    total_questions = 0
    
    for category, entries in categories_data.items():
        if not entries or category not in questions_distribution:
            continue
            
        print(f"🏷️ Processing category: {category}")
        
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
    
    print(f"\n📊 Benchmark dataset created:")
    print(f"   - Categories: {len(dataset['kategorien'])}")
    print(f"   - Total questions: {total_questions}")
    
    return dataset

def main():
    """Main function: Creates benchmark questions from lexicon entries."""
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🚀 Starting benchmark question generator")
    print(f"📂 Input directory: {INPUT_DIR} (all lexikon_*.json files)")
    print(f"📂 Output directory: {OUTPUT_DIR}")
    print(f"📄 Output file: {OUTPUT_FILENAME}")
    print(f"🔧 API type: {api_name}")
    print(f"🔧 Server: {server_url}")
    print(f"🔧 Model: {model_name}")
    print(f"🎯 Max total questions: {MAX_TOTAL_QUESTIONS}")
    print(f"🎯 Small categories: {MIN_QUESTIONS_PER_CATEGORY}-{MAX_QUESTIONS_PER_CATEGORY} questions")
    print(f"🎯 Question type distribution: {int(QUESTION_TYPE_DISTRIBUTION['definition']*100)}% definitions, {int(QUESTION_TYPE_DISTRIBUTION['transfer']*100)}% transfer")
    print(f"🌍 Language agnostic: ✅")
    
    # Test API connection
    if not check_api_connection():
        print("❌ API connection failed. Processing aborted.")
        return
    
    # Load lexicon entries by categories
    categories_data = load_lexikon_files_by_category(INPUT_DIR)
    
    if not categories_data:
        print("❌ No lexicon entries found!")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating benchmark questions")
    print(f"{'='*60}")
    
    # Create benchmark dataset
    benchmark_dataset = create_benchmark_dataset(categories_data)
    
    # Save the dataset
    output_path = Path(OUTPUT_DIR) / OUTPUT_FILENAME
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"🎉 Benchmark questions successfully created!")
    print(f"📊 Statistics:")
    print(f"   - API used: {api_name}")
    print(f"   - Server: {server_url}")
    print(f"   - Model: {model_name}")
    
    total_questions = sum(cat["anzahl_fragen"] for cat in benchmark_dataset["kategorien"])
    total_entries = sum(len(entries) for entries in categories_data.values())
    
    print(f"   - Categories: {len(benchmark_dataset['kategorien'])}")
    print(f"   - Benchmark questions: {total_questions} (max. {MAX_TOTAL_QUESTIONS})")
    print(f"   - Available entries: {total_entries}")
    print(f"   - All lexikon_*.json files processed: ✅")
    print(f"   - Language agnostic: ✅")
    print(f"   - File saved: {output_path}")
    
    # Show example questions
    print(f"\n📋 Example benchmark questions:")
    for category_data in benchmark_dataset["kategorien"][:2]:  # First 2 categories
        if category_data["fragen"]:
            example = category_data["fragen"][0]
            print(f"   [{category_data['kategorie']}] {example['id']}")
            print(f"   Question: {example['frage']}")
            print(f"   Answer: {example['antwort'][:100]}...")
            print()

if __name__ == "__main__":
    # Show configuration
    api_name = "OpenAI-API" if USE_OPENAI_API else "Ollama-API"
    server_url = OPENAI_BASE_URL if USE_OPENAI_API else OLLAMA_SERVER_URL
    model_name = OPENAI_MODEL_NAME if USE_OPENAI_API else OLLAMA_MODEL_NAME
    
    print(f"🔧 CONFIGURATION (from central config):")
    print(f"   - API type: {api_name}")
    print(f"   - Server: {server_url}")
    print(f"   - Model: {model_name}")
    print(f"   - Random selection: ✅ Activated")
    print(f"   - Language agnostic: ✅ Activated")
    
    # Main execution
    main()