
# OpenTuneWeaver

Dieses Projekt ermöglicht ein vereinfachtes Finetuning eines OpenSource-LLMs. Dabei wird zunächst eine RAW-PDF hochgeladen, dann in Q&A überführt und in ein passendes Finetuning-Format überführt. Danach wird mittels ausgewählten Mitteln ein ausgewähltes OpenSource-LLM feinabgestimmt und zum Download (auch als gguf) bereitgestellt. Alles in einer anwenderfreundlichen UI.

## Projektstruktur

- `data/`: Verzeichnis für Rohdaten wie PDFs und konvertierte TXT-Dateien.
- `models/`: Beinhaltet trainierte Modelle und Adapter. Unterteilt in `finetuned_models/` für trainierte Modelle und `adapters/` für (Q)LORA-Adapter oder ähnliche.
- `scripts/`: Enthält Skripte für Datenvorbereitung (`data_preparation.py`), Finetuning (`finetuning.py`) und Generierung von Q&A oder Chats (`qa_chat_generation.py`).
- `gradio_app/`: Hauptverzeichnis für die Gradio-App. Die `app.py` dient als Hauptdatei für die App, und `components/` beinhaltet wiederverwendbare Komponenten für die UI.
- `requirements.txt`: Liste aller benötigten Python-Pakete für das Projekt.
- `README.md`: Diese Datei. Bietet eine Übersicht und Anleitung für das Projekt.

## Installation

Um dieses Projekt zu verwenden, wird empfohlen, eine isolierte Python-Umgebung mit Conda zu erstellen und zu verwenden. Folgen Sie diesen Schritten, um das Projekt einzurichten:

1. **Klonen des Repositories:**
   Öffnen Sie ein Terminal und klonen Sie das Repository mit dem folgenden Befehl:

   ```bash
   git clone https://github.com/ProfEngel/OpenTuneWeaver
   cd OpenTuneWeaver
   ```

2. **Erstellen einer Conda-Umgebung:**
   Erstellen Sie eine neue Conda-Umgebung namens `opentuneweaver` mit Python 3.11:

   ```bash
   conda create -n opentuneweaver python=3.11
   conda activate opentuneweaver
   ```

3. **Installieren der Abhängigkeiten:**
   Installieren Sie die erforderlichen Python-Pakete:

   ```bash
   pip install -r requirements.txt
   ```

## Verwendung

Um die Gradio-App zu starten, navigieren Sie zum `gradio_app/` Verzeichnis und führen Sie das `app.py` Skript aus:

```bash
cd gradio_app
python app.py
```

Die App sollte nun laufen und über einen lokalen Server auf Ihrem Computer zugänglich sein. Öffnen Sie den in der Konsole angegebenen Link in Ihrem Browser, um die App zu verwenden.

## Datenvorbereitung

Legen Sie Ihre PDF-Dateien im `data/` Verzeichnis ab. Verwenden Sie das entsprechende Skript im `scripts/` Verzeichnis oder die Gradio-UI, um die Daten in das erforderliche Format zu konvertieren.

## Finetuning

Wählen Sie in der Gradio-UI das gewünschte Modell und den Datensatz für das Finetuning aus. Das entsprechende Skript im `scripts/` Verzeichnis wird verwendet, um das Modell mit Ihrem Datensatz zu finetunen.

## Generierung

Nach dem Finetuning können Sie die Q&A- oder Chat-Generierungsfunktion nutzen, um auf Basis der verarbeiteten Texte Fragen zu beantworten oder einen Chat zu generieren.

## Beitrag

Wir freuen uns über Beiträge von jedem! Wenn Sie einen Fehler finden, eine Funktion anfragen oder einen Pull-Request einreichen möchten, fühlen Sie sich frei, das zu tun.

## Roadmap? 🛣️
Was als nächstes implementiert wird, habe ich auf meiner Roadmap dargestellt. Das findet man in meiner [Documentation](https://github.com/ProfEngel/OpenTuneWeaver/wiki/Roadmap).

# <iframe style="width:100%;height:auto;min-width:600px;min-height:400px;" src="https://star-history.com/embedsecret=Z2hwX0JRaFJPZXkxcUpibDJ6Q1dYS052YUVleEJtRG5YaDFMcjJlMw==#ProfEngel/OpenTuneWeaver&Timeline" frameBorder="0"></iframe>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ProfEngel/OpenTuneWeaver&type=Timeline)](https://star-history.com/#ProfEngel/OpenTuneWeaver&Timeline)


## Lizenz📜

Dieses Projekt ist nach der [MIT](LICENSE)-Lizenz lizenziert. Schau in das Lizenzfile für Details und frag gegebenenfalls ChatGPT oder Dein LLM was das bedeutet.📄

Erstellt von [Prof. Dr. Mathias Engel](https://github.com/ProfEngel)  - Lasst uns Open-Source LLM für alle Anwendungszwecke spezialisieren! 💪

## Credits

Um diese vereinfachte App zu ermöglichen, wurden Inhalte aus folgenden Quellen (in Teilen) verwendet.

## Related Papers

Veröffentlichungen zu diesem Repo werde ich hier verlinken.
