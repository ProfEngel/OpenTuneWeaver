import os
from pdfminer.high_level import extract_text
from pathlib import Path

def pdf_to_txt(pdf_path, txt_path):
    """
    Konvertiert eine PDF-Datei in eine TXT-Datei.
    
    Args:
        pdf_path (str): Der Pfad zur PDF-Datei.
        txt_path (str): Der Pfad, unter dem die TXT-Datei gespeichert werden soll.
    """
    text = extract_text(pdf_path)
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text)

def process_pdfs(input_folder, output_folder):
    """
    Verarbeitet alle PDFs in einem Verzeichnis, benennt sie um und speichert die extrahierten Texte als TXT.

    Args:
        input_folder (str): Verzeichnis mit den PDF-Dateien.
        output_folder (str): Verzeichnis, in dem die TXT-Dateien gespeichert werden sollen.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(Path(input_folder).glob("*.pdf"))
    for i, pdf_file in enumerate(pdf_files, start=1):
        new_pdf_name = f"raw{i:02d}.pdf"  # Erstellt Namen wie raw01, raw02, ...
        new_pdf_path = Path(input_folder) / new_pdf_name
        os.rename(pdf_file, new_pdf_path)  # PDF umbenennen und verschieben
        
        txt_filename = f"raw{i:02d}.txt"
        txt_path = Path(output_folder) / txt_filename
        pdf_to_txt(new_pdf_path, txt_path)
        print(f"{new_pdf_name} wurde zu {txt_filename} konvertiert und unter {output_folder} gespeichert.")

if __name__ == "__main__":
    input_folder = "../data"
    output_folder = "../data/txt"
    process_pdfs(input_folder, output_folder)
