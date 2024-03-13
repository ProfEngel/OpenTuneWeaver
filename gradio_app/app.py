import gradio as gr
from model_functions import process_input, finetune_model, generate_qa, generate_chat

def process_files(files):
    # Verarbeiten der hochgeladenen Dateien, konvertieren zu TXT usw.
    return process_input(files)

def finetune(llm, dataset):
    # Finetuning des LLM basierend auf der Auswahl und dem bereitgestellten Datensatz
    return finetune_model(llm, dataset)

def generate_output(mode, input_text):
    # Generiere Q&A oder Chat basierend auf dem Eingabetext
    if mode == "Q&A":
        return generate_qa(input_text)
    else:
        return generate_chat(input_text)

# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("### DeinProjekt")
    with gr.Tab("Datenvorbereitung"):
        with gr.Row():
            file_input = gr.Files(label="Upload PDFs")
            preprocess_btn = gr.Button("Konvertieren")
        preprocess_btn.click(process_files, inputs=file_input, outputs=gr.Textbox(label="Status"))

    with gr.Tab("Generierung"):
        with gr.Row():
            mode_select = gr.Radio(choices=["Q&A", "Chat"], label="Modus")
            text_input = gr.Textbox(label="Frage/Eingabe")
            generate_btn = gr.Button("Generieren")
        generate_btn.click(generate_output, inputs=[mode_select, text_input], outputs=gr.Textbox(label="Antwort"))
        
    with gr.Tab("Finetuning"):
        with gr.Row():
            model_select = gr.Radio(choices=["LLM1", "LLM2"], label="Wähle ein Modell")
            finetune_btn = gr.Button("Finetune")
        finetune_btn.click(finetune, inputs=[model_select, file_input], outputs=gr.Textbox(label="Finetuning Status"))
        
app.launch()
