# app.py
import gradio as gr
from model_functions import save_pdf_files, pdf_to_txt, finetune_model, generate_qa, generate_chat

def upload_files(files):
    # Speichert die PDF-Dateien und gibt die Namen der gespeicherten Dateien zurück
    save_msg, filenames = save_pdf_files(files)
    return save_msg, "\n".join(filenames)

def convert_files():
    # Konvertiert die gespeicherten PDF-Dateien in TXT
    convert_msg = pdf_to_txt()
    return convert_msg

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
    gr.Markdown("### OpenTuneWeaver")
    with gr.Tab("Datenvorbereitung"):
        file_input = gr.Files(label="Upload PDF", file_types=["image", "video", "pdf"], file_count="multiple")
        upload_button = gr.Button("PDFs hochladen")
        uploaded_files_output = gr.Text(label="Hochgeladene Dateien")
        convert_button = gr.Button("Konvertieren")
        convert_status_output = gr.Text(label="Konvertierungsstatus")
        
        upload_button.click(fn=upload_files, inputs=file_input, outputs=[convert_status_output, uploaded_files_output])
        convert_button.click(fn=convert_files, inputs=None, outputs=convert_status_output)




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
