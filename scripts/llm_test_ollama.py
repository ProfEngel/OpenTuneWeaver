#https://www.markhneedham.com/blog/2023/10/18/ollama-hugging-face-gguf-models/
#Zuerst aus dem leo-mistral-gguf Modell ein Ollama-Modell erstellen. Siehe dazu Link oben. Das hier einbinden

#dann....

from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
  model="leo-mistral",
  messages=[
    {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
    #{"role": "user", "content": "Who won the world series in 2020?"},
    #{"role": "assistant", "content": "The LA Dodgers won in 2020."},
    {"role": "user", "content": "Warum ist der Himmel blau?"}
  ]
)
print(response.choices[0].message.content)