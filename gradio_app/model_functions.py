# model_functions.py
from pathlib import Path
import os
from pdfminer.high_level import extract_text

def save_pdf_files(files, save_path="../data"):
    """
    Speichert hochgeladene PDF-Dateien in einem Verzeichnis.
    Args:
        files: Eine Liste von hochgeladenen Dateien.
        save_path (str): Der Pfad, unter dem die Dateien gespeichert werden sollen.
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    for i, file in enumerate(files, start=1):
        file_path = os.path.join(save_path, f"raw{i:02d}.pdf")
        with open(file_path, "wb") as f:
            # Schreibe den Inhalt der Datei direkt, anstatt getbuffer() zu verwenden
            f.write(file.read())  # Verwende .read() für NamedString-Objekte
            
    return f"{len(files)} PDF-Datei(en) gespeichert."

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
