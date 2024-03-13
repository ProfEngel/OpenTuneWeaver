# DeinProjekt

DeinProjekt ist eine GradioApp, die darauf abzielt, eine Schnittstelle für verschiedene Funktionen zu bieten, einschließlich der Verarbeitung und Analyse von Büchern/Papers, der Generierung von Q&A und Chat basierend auf den verarbeiteten Texten sowie der Vorbereitung von Trainings- und Testdatensätzen für das Fine-Tuning von Large Language Models (LLMs).

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
git clone <repository-url>
cd DeinProjekt
pip install -r requirements.txt
