import os
import shutil
import gradio as gr

# Bestimmen Sie den absoluten Pfad des data-Ordners relativ zu diesem Skript
current_directory = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(current_directory, 'data')

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

def save_pdf(file_info):
    if file_info is not None:
        # Extrahieren Sie den Dateinamen
        file_name = file_info.name if hasattr(file_info, 'name') else os.path.basename(file_info)
        # Konstruieren Sie den Zielpfad im `data`-Ordner
        save_path = os.path.join(current_directory, 'data')
        # Verschieben Sie die Datei zum Zielort
        shutil.move(file_info, save_path)
        return f"Datei erfolgreich in {save_path} gespeichert."
    else:
        return "Keine Datei hochgeladen."

iface = gr.Interface(fn=save_pdf, inputs=gr.File(label="PDF-Datei hochladen"), outputs="text")
iface.launch()
