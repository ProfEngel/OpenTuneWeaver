import os
import shutil
import gradio as gr

# Bestimmen des übergeordneten Verzeichnisses und des Zielordners `data`
parent_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_folder = os.path.join(parent_directory, 'data')

# Stellen Sie sicher, dass der data-Ordner existiert
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

def upload_files(files):
    saved_files_paths = []
    for file_info in files:
        if file_info is not None:
            file_name = os.path.basename(file_info.name)
            save_path = os.path.join(data_folder, file_name)
            shutil.move(file_info.name, save_path)
            saved_files_paths.append(save_path)
    if saved_files_paths:
        return f"Dateien erfolgreich gespeichert: {', '.join(saved_files_paths)}"
    else:
        return "Keine Dateien hochgeladen."

# Gradio Interface mit Blocks
with gr.Blocks() as app:
    gr.Markdown("### OpenTuneWeaver")
    with gr.Tab("Datenvorbereitung"):
        with gr.Row():
            file_input = gr.File(label="PDF- oder DOCX-Datei hochladen", file_types=["pdf", "docx"])
            submit_button = gr.Button("Hochladen")
        file_output = gr.TextArea()

    # Aktion festlegen
    submit_button.click(fn=upload_files, inputs=file_input, outputs=file_output)

app.launch()
