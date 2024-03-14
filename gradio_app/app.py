# app.py
import os
import shutil
import gradio as gr
from model_functions import save_pdf, pdf_to_txt, finetune_model, generate_qa, generate_chat

def convert_files():
    """Konvertiert die zuvor gespeicherten PDF-Dateien in TXT."""
    convert_msg = pdf_to_txt()
    return convert_msg

def generate_output(mode, input_text):
    # Generiere Q&A oder Chat basierend auf dem Eingabetext
    if mode == "Q&A":
        return generate_qa(input_text)
    else:
        return generate_chat(input_text)
    
def finetune(llm, dataset):
    # Finetuning des LLM basierend auf der Auswahl und dem bereitgestellten Datensatz
    return finetune_model(llm, dataset)

# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("### OpenTuneWeaver")
    with gr.Tab("Datenupload und -konvertierung"):
        gr.Markdown("""### Datenupload und -konvertierung
        
        Laden Sie hier eine oder mehrere PDF-Datei(en) hoch. Nach dem Hochladen der Datei(en) wird/werden diese automatisch in den `data`-Ordner gespeichert. Klicken Sie anschließend auf "Konvertieren", um die PDF(s) in eine/mehrere Textdatei(en) zu konvertieren. Der Konvertierungsstatus wird unten angezeigt. Sobald dieser abgeschlossen ist, kann die Anpassung im nächsten Reiter beginnen.""")
        gr.Interface(save_pdf, inputs=gr.File(label="PDF-Datei hochladen", file_count="multiple"), outputs=None)
        #gr.Interface(inputs=gr.File(label="PDF-Datei hochladen", file_count="multiple"), outputs=None)
        convert_button = gr.Button("Konvertieren")
        convert_status_output = gr.Textbox(label="Konvertierungsstatus")
        convert_button.click(fn=convert_files, inputs=None, outputs=convert_status_output)



    with gr.Tab("Datensatzanpassung"):
        with gr.Row():
            mode_select = gr.Radio(choices=["Q&A", "Chat"], label="Modus")
            text_input = gr.Textbox(label="Frage/Eingabe")
            generate_btn = gr.Button("Generieren")
        generate_btn.click(generate_output, inputs=[mode_select, text_input], outputs=gr.Textbox(label="Antwort"))
        
    with gr.Tab("Finetuning"):
        with gr.Row():
            model_select = gr.Radio(choices=["LLM1", "LLM2"], label="Wähle ein Modell")
            finetune_btn = gr.Button("Finetune")
            
        finetune_btn.click(finetune, inputs=[model_select, text_input], outputs=gr.Textbox(label="Finetuning Status"))

    with gr.Tab("Playground"):
        with gr.Row():
            model_select = gr.Radio(choices=["LLM1", "LLM2"], label="Wähle ein Modell")
            finetune_btn = gr.Button("Finetune")
            
        finetune_btn.click(finetune, inputs=[model_select, text_input], outputs=gr.Textbox(label="Finetuning Status"))
   


app.launch(share=True)
