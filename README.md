# FineTuneLLM

Dieses Projekt ermöglicht eine vereinfachtes Finetuning eines OpenSource-LLMs. Dabei wird zunächst eine RAW-PDF hochgeladen, dann in Q&A überführt und in ein passendes Finetuning-Format überführt. Danach wird mittels ausgewählten Mitteln ein ausgewähltes OpenSource-LLM feinabgestimmt und zum Download (auch als gguf) bereitgestellt. Alles in einer anwenderfreundlichen UI.

## Projektstruktur

- `data/`: Verzeichnis für Rohdaten wie PDFs und konvertierte TXT-Dateien.
- `models/`: Beinhaltet trainierte Modelle und Adapter. Unterteilt in `finetuned_models/` für trainierte Modelle und `adapters/` für (Q)LORA-Adapter oder ähnliche.
- `scripts/`: Enthält Skripte für Datenvorbereitung (`data_preparation.py`), Finetuning (`finetuning.py`) und Generierung von Q&A oder Chats (`qa_chat_generation.py`).
- `gradio_app/`: Hauptverzeichnis für die Gradio-App. Die `app.py` dient als Hauptdatei für die App, und `components/` beinhaltet wiederverwendbare Komponenten für die UI.
- `requirements.txt`: Liste aller benötigten Python-Pakete für das Projekt.
- `README.md`: Diese Datei. Bietet eine Übersicht und Anleitung für das Projekt.

## Installation

Um dieses Projekt zu verwenden, klonen Sie das Repository und installieren Sie die erforderlichen Abhängigkeiten:

```bash
git clone https://github.com/ProfEngel/FineTuneLLM
cd FineTuneLLM
pip install -r requirements.txt
