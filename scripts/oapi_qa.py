import PyPDF2
import json
import requests

def extract_text_from_pdf(file_path):
    pdf_file_obj = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page_num]
        text += page_obj.extract_text()
    pdf_file_obj.close()
    return text

def split_text_into_chunks(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def submit_to_ollama_api(prompt, retries=3):
    url = 'http://localhost:11434/api/chat'
    detailed_instruction = (
        "Basierend auf dem folgenden Text, formuliere bitte klare und präzise Frage-Antwort-Paare. "
        "Die Fragen sollten mit 'Im Kontext ...' beginnen, um den Bezug zum Text deutlich zu machen. "
        "Antworten sollten direkt aus dem Text abgeleitet und wortwörtlich übernommen werden. "
        "Ziel ist es, die Informationen im Text durch diese Q&A zugänglich zu machen. "
        "Hier ist ein Beispiel, wie deine Antworten aussehen sollten:\n\n"
        "Frage: Im Kontext von Wirtschaft, was versteht man unter einem Unternehmen?\n"
        "Antwort: Ein Unternehmen ist eine Organisation oder ein Geschäft, das kommerzielle, industrielle oder professionelle Aktivitäten ausführt, um Waren zu produzieren oder Dienstleistungen gegen Gewinn anzubieten.\n"
    )
    payload = {
        "model": "mistral",
        "temperature": 0.5,
        "stream": False,
        "messages": [
            {"role": "system", "content": detailed_instruction},
            {"role": "user", "content": prompt}
        ]
    }
    for i in range(retries):
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                message_str = response.content.decode('utf-8')
                message_dict = json.loads(message_str)
                if 'content' in message_dict.get('message', {}):
                    print("Aktuelle Q&A:")
                    print(message_dict['message']['content'])
                    print("-" * 80)  # Eine Trennlinie für bessere Lesbarkeit
                    return message_dict['message']['content']
                else:
                    print("No valid content found in the response.")
            else:
                print(f"Attempt {i + 1} failed with status code {response.status_code}. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
    print("Max retries exceeded. Skipping this prompt.")
    return None

text = extract_text_from_pdf('test.pdf')
chunks = split_text_into_chunks(text, 512)

responses = []
for chunk in chunks:
    response = submit_to_ollama_api(chunk)
    if response:
        responses.append(response)
        print("Aktuelle Q&A:")
        print(response)
        print("-" * 80)  # Eine Trennlinie für bessere Lesbarkeit

# Optional: Speichern der Antworten in einer Datei oder weitere Verarbeitung
with open('ollama_responses2.json', 'w', encoding='utf-8') as f:
    json.dump(responses, f, ensure_ascii=False, indent=4)
