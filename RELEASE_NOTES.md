# Release v0.1.0 - The Genesis 🧬

We are thrilled to announce the first official release of **OpenTuneWeaver (v0.1.0)**! This milestone marks the transition from a research prototype to a robust, curatable, all-in-one document pipeline for LLM dataset generation.

## 🚀 Key Features

- **End-to-End Automation**: Transform raw PDFs, DOCX, and images into structured Wikis and InstructQA datasets.
- **Semantic Wiki Chunking**: Deep knowledge extraction using context-aware synthesis instead of simple text splitting.
- **Bidirectional QA Generation**: Core innovation creating symmetrical knowledge mappings (A -> Q and Q -> A).
- **Integrated Viewer Environment**: A beautiful, dark-mode UI to review, edit, and curate your datasets before training.
- **Hardware Agnostic**: Fully API-driven—connect to Ollama, OpenAI, or any endpoint without needing a massive local GPU.

## 🆕 What's New in this Release

- **Project Vision & Personas**: Added "Who is this for?" section targeting LLM Developers and Enterprise AI teams.
- **Enhanced Roadmap**:
    - **Multi-Dataset Management**: Support for independent projects and dataset silos.
    - **Advanced Export Formats**: Pre-planning for Alpaca, ChatML, and ShareGPT conversion layers.
- **Documentation Overhaul**: Complete English README update for global accessibility.

## 🛠 Installation

You can run OpenTuneWeaver via Docker or directly on Ubuntu:

```bash
docker run -d -p 3030:8080 -v opentuneweaver_data:/app/pipeline/OUTPUT --name opentuneweaver opentuneweaver:latest
```

## 🙏 Acknowledgments

A huge thank you to the open-source community and the tools that power OpenTuneWeaver: Gradio, Docling, Ollama, and Marker. Special thanks to **MWK Baden-Württemberg** and **Stifterverband Deutschland** for supporting this research.

---

**Full Changelog**: [v0.1.0-changes](https://github.com/ProfEngel/OpenTuneWeaver/commits/v0.1.0)

**Tags**: `llm` `dataset-generation` `synthetic-data` `ai` `machine-learning` `nlp` `semantic-chunking` `pdf-processing` `qa-generation` `gradio` `research-tools`
