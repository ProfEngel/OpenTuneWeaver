# OpenTuneWeaver

**Semantisch-strukturierte, kuratierbare LLM-Finetuning-Pipeline**

OpenTuneWeaver (OTW) ist eine Open-Source-Plattform, die aus beliebigen Dokumenten (PDF, DOCX, etc.) logisch strukturierte "Wiki"-Einträge, InstructQA- und Benchmark-Datensätze erzeugt und in eine komplette Fine-Tuning- und Evaluierungs-Pipeline überführt. Besondere Merkmale:

- Semantisches Chunking: Unterüberschrift + zugehöriger Inhalt bilden Wissenseinheiten.
- Automatisierte InstructQA-Generierung mit 5 Fragetypen und Markdown-Erhalt.
- Automatisierte Benchmark-Erstellung mit Ground-Truth und Schwierigkeitsgraden.
- Kuratierbare Viewer-Umgebung für Mensch-in-der-Schleife-Review.
- GPU-auto-optimiertes Finetuning (LoRA/QLoRA), Multi-Format-Export (LoRA, GGUF).
- End-to-End-Workflow: Dokumentenkonvertierung → Datensatzgenerierung → Training → Evaluation.

## Installation auf Runpod

Beispiel mit dem Docker-Image `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`:

```bash
# Auswahl des Runpod-Containers
docker pull runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

docker run --gpus all -it \
  -v $(pwd):/workspace \
  runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 /bin/bash

# Im Container
cd /workspace

# Repository klonen
git clone https://github.com/ProfEngel/OpenTuneWeaver.git

# Setup-Skript kopieren und ausführen
cp OpenTuneWeaver/setup_runpod_direct.sh .
chmod +x setup_runpod_direct.sh
./setup_runpod_direct.sh
```

## Schnellstart

1. Dateien im `pipeline/modules/01_convert/UPLOAD/` ablegen.
2. `python3 run_pipeline.py --auto --mode full` ausführen.
3. Ergebnisse (Markdown, Modelle, Benchmarks) im `pipeline/data/OUTPUT/` finden.
4. Optional: Gradio-UI starten mit `python3 ui/app.py`.

## Lizenz

Apache License 2.0