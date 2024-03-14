# model_functions.py
from pathlib import Path
import shutil
import os
from pdfminer.high_level import extract_text

# Pfad des übergeordneten Ordners von `gradio_app`
parent_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_folder = os.path.join(parent_directory, 'data')

# Stellen Sie sicher, dass der data-Ordner existiert
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

def save_pdf(file_infos):
    saved_files = []  # Eine Liste, um die Namen der erfolgreich gespeicherten Dateien zu sammeln
    
    for file_info in file_infos:

        if file_info is not None:
            # Der Dateiname bleibt gleich, da `shutil.move` den Originalnamen beibehält
            file_name = os.path.basename(file_info)
            
            # Pfad, wohin die Datei verschoben wird (der `data`-Ordner)
            save_path = data_folder
            
            # Datei in den `data`-Ordner verschieben
            shutil.move(file_info, save_path)
        
    if saved_files:
        return f"Datei(en) erfolgreich in {data_folder} gespeichert: {', '.join(saved_files)}"
    else:
        return "Keine Datei hochgeladen."

def pdf_to_txt(input_folder="../data", output_folder="../data/txt"):
    """
    Konvertiert alle PDF-Dateien in einem Verzeichnis in TXT-Dateien.
    Args:
        input_folder (str): Das Verzeichnis mit den PDF-Dateien.
        output_folder (str): Das Verzeichnis, in dem die TXT-Dateien gespeichert werden.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    pdf_files = list(Path(input_folder).glob("*.pdf"))
    if not pdf_files:
        return "Keine PDF-Dateien zum Konvertieren gefunden."
    
    for i, pdf_file in enumerate(pdf_files, start=1):
        text = extract_text(str(pdf_file))
        txt_filename = f"raw{i:02d}.txt"
        txt_path = Path(output_folder) / txt_filename
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)
    
    return f"{len(pdf_files)} PDF-Datei(en) zu TXT konvertiert."

# Die restlichen Funktionen bleiben unverändert
def finetune_model(llm, dataset):
    return "Modell finetuned"

def generate_qa(input_text):
    return "Q&A Antwort für: " + input_text

def generate_chat(input_text):
    return "Chat-Antwort für: " + input_text
