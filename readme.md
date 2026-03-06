# OpenTuneWeaver 🧬

![GitHub stars](https://img.shields.io/github/stars/ProfEngel/OpenTuneWeaver?style=social)
![GitHub forks](https://img.shields.io/github/forks/ProfEngel/OpenTuneWeaver?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/ProfEngel/OpenTuneWeaver?style=social)
![GitHub repo size](https://img.shields.io/github/repo-size/ProfEngel/OpenTuneWeaver)
![GitHub language count](https://img.shields.io/github/languages/count/ProfEngel/OpenTuneWeaver)
![GitHub top language](https://img.shields.io/github/languages/top/ProfEngel/OpenTuneWeaver)
![GitHub last commit](https://img.shields.io/github/last-commit/ProfEngel/OpenTuneWeaver?color=red)
[![Sponsor](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=ff69b4)](https://github.com/sponsors/ProfEngel)
[![YouTube](https://img.shields.io/badge/YouTube-MatMaxEngel-red?logo=youtube&logoColor=white)](https://www.youtube.com/user/MatMaxEngel)
[![Website](https://img.shields.io/badge/Website-opentuneweaver.com-blue?logo=googlechrome&logoColor=white)](https://opentuneweaver.com/)

<div align="center">
  <img src="assets/otw_logo_banner.png" alt="OpenTuneWeaver Logo" width="900">
</div>

![OTW pipeline overview](assets/8.png)
*Pipeline status and progress.*

**OpenTuneWeaver is a semantically-structured, curatable all-in-one API-based document pipeline that automatically creates structured wiki entries and diverse InstructQA datasets from any raw data (PDF, DOCX, etc.).** The system revolutionizes how knowledge is extracted by providing **semantic chunking**, **bidirectional AI Q&A generation**, and a **clean UI** without requiring expensive local GPUs or technical expertise.

<div align="right">
  <img src="assets/mwk_logo_w2.png" alt="Ministry of Science, Research and Arts Logo" height="60">
  <img src="assets/stifterverband_logo.jpg" alt="Stifterverband Logo" height="60">
</div>

This project is part-funded by the **Ministry of Science, Research and Arts Baden-Württemberg (MWK)** and **Stifterverband Deutschland** as part of digital [Fellowship-Program](https://www.stifterverband.org/bwdigifellows/2024_engel_leiblein).

![OpenTuneWeaver Demo](assets/OTW_demo.gif)

## 💖 Support OpenTuneWeaver

Help us democratize AI development for education and research! OpenTuneWeaver is completely free to use, and we want to keep it that way. Your support enables us to continue building accessible AI tools without any paywalls.

[![Sponsor](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=ff69b4)](https://github.com/sponsors/ProfEngel)

**[Become a sponsor and join our mission!](https://github.com/sponsors/ProfEngel)** 🚀

> **🚀 Get Started with OpenTuneWeaver**
> 
> **✅ 100% Free**  
> We have eliminated all commercial licensing. You no longer need to pay for an enterprise tier! Support us voluntarily with donations.
>  
> **💻 Work with Any API**  
> No more 24GB GPUs needed locally. Connect to Ollama, OpenAI, or any endpoint, and build datasets easily.
***
***

## Key Features 🚀

- 🔄 **End-to-End Automation**: From complex PDFs with tables straight to QA Dataset ready for AI consumption
- 🧠 **Bidirectional Knowledge Generation**: Core innovation creating *Inverse QA pairs (A -> Q)* ensuring models deeply map concepts and relationships symmetrically.
- 📚 **Automatic Dataset Creation**: Automatically builds Wiki entries and versatile InstructQA formats.
- 🎨 **Curatable Viewer Environment**: Clean, completely dark-mode aligned responsive interface allowing manual QA and fixes safely.
- 🌐 **Hardware Agnostic via API Config**: Works anywhere! Fully operational with `docling` extracting data, while external APIs (like Docker-bound Ollama endpoints or OpenAI) do the LLM reasoning processing.
- 🔍 **VLM Integration**: Seamless integration of Vision-Language-Models to recognize and describe embedded images.

### The OpenTuneWeaver Innovation 🎯

**Why is OpenTuneWeaver different?** Traditional LLM dataset preparation is often a messy, fragmented, and highly technical process. OpenTuneWeaver revolutionizes this by offering a **seamless, end-to-end pipeline** that brings order to chaos. 

It takes any unstructured raw document—complete with complex tables, multi-column layouts, and embedded images—and intelligently transforms it into highly structured, interconnected knowledge. Utilizing advanced vision models to "see" your documents, the pipeline performs **Semantic Wiki Chunking** to synthesize context-rich lexicon entries rather than blindly chopping text into pieces. 

From these synthesized wikis, it autonomously generates diverse InstructQA datasets. This includes our core innovation of **Bidirectional Knowledge Generation**, ensuring that your model learns relational concepts symmetrically.

**Full Control with the Built-in Viewer:**
You are never isolated from your data. Instead of digging through raw `.jsonl` files in a code editor, OpenTuneWeaver features an **integrated, dark-mode Viewer Environment**. You can directly read, review, edit, and safely save every generated markdown file, lexicon wiki, and QA pair right inside the application, ensuring maximum dataset quality before any training begins.

![OpenTuneWeaver Viewer Demo](assets/OTW_viewer_demo.gif)

***

## How to Install 🚀

### Option 1: Docker (Highly Recommended)
We provide a highly optimized Docker image that handles all PyTorch and `docling` dependencies for you cleanly.

1. **Clone the repository:**
```bash
git clone https://github.com/profengel/opentuneweaver.git
cd opentuneweaver
```

2. **Build the container locally:**
```bash
docker build -t opentuneweaver .
```

3. **Run the container (Requires port 3030):**
```bash
# Uses host-gateway to correctly bridge API requests to your local Ollama/LM Studio or APIs like OpenRouter/OpenAI
docker run -d -p 3030:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v opentuneweaver_data:/app/pipeline/OUTPUT \
  --name opentuneweaver \
  --restart always \
  opentuneweaver:latest
```
Access the application on your browser at `http://localhost:3030`.

#### Troubleshooting Docker Build on Linux/Ubuntu/WSL2
If the `docker build` fails during the `apt-get update` step with errors like `Temporary failure resolving 'deb.debian.org'`, it is usually a DNS issue inside the Docker build container. You can fix this by using your host's network during the build:
```bash
docker build --network host -t opentuneweaver .
```
*(Alternatively, configure your `/etc/docker/daemon.json` to use public DNS servers like `8.8.8.8`.)*

### Option 2: Local Installation (Linux/Windows/Mac)

1. Clone the repository
```bash
git clone https://github.com/ProfEngel/OpenTuneWeaver.git
cd OpenTuneWeaver
```

2. Create a Virtual Environment & Install Requirements
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Start OpenTuneWeaver
```bash
python app.py
```

***

## What's Next? 🌟

**Short to medium-term roadmap:**
- 🌍 **Multilingual Support**: German, Spanish, French, additional languages
- 🔬 **ArXiv Publication**: Publishing the foundational research paper for OpenTuneWeaver's methodology on ArXiv
- 📟 **Real LiveView**: Building a genuine, Container-aware Live Terminal directly into the UI for full transparency on background processes
- 🗄️ **Direct VectorDB Export**: One-click integrations to export generated datasets straight to ChromaDB, Pinecone, or Qdrant for immediate RAG deployments
- 🕸️ **Web Ingestion**: Direct support for URLs and Web-Scraping to convert entire websites into curatable Wiki/QA knowledge
- 🤖 **Agentic QA Evaluation**: Implementing an "LLM-as-a-Judge" pipeline to automatically score and filter generated QA pairs for maximum quality
- 🔗 **MCP-Server Integration**: OpenTuneWeaver as MCP-Server for direct chat integration and automation pipelines
- 🎥 **YouTube Tutorials**: Comprehensive video tutorials on [MatMaxEngel YouTube Channel](https://www.youtube.com/user/MatMaxEngel) covering usage guides

***

## Media Coverage & Interviews 📰

OpenTuneWeaver and our research on AI in education have gained significant media attention. Here are recent interviews and articles featuring Prof. Dr. Mathias Engel and the project:

### Recent Press Coverage

**⚡ [Lehr/Lernkonferenz 2025 - "Erprobung eines MoE und MultiAgenten – Chatbot als KI-Tutor für die Lehre"](https://www.lehrlernkonferenz-2025.de/programm)**  
Lightning talk exploring the implementation of Mixture of Experts (MoE) and multi-agent chatbot systems as AI tutors in educational settings.

**🎤 [HAWAII der GHD - "Level up! KI-Tutor „Käpsele" und trainiertes Sprachmodell „Hölderlin" im Multiplayer-Modus"](https://www.hochschuldidaktik.net/hawaii-25)**  
Conference presentation demonstrating advanced AI tutoring systems in multiplayer mode.

**📰 [VDI Nachrichten - "Professor Chatbot hilft den Studierenden"](https://www.vdi-nachrichten.com/karriere/studium/professor-chatbot/)**  
Technical magazine article exploring how universities increasingly deploy artificial intelligence to enhance teaching quality.

**📄 [Controlling & Management Review - "Generative KI im Controlling praktisch umsetzen"](https://www.springerprofessional.de/generative-ki-im-controlling-praktisch-umsetzen/51394852)**  
Reviewed paper discussing practical implementation of generative AI in controlling.

**📰 [Nürtinger Zeitung - "Wie künstliche Intelligenz beim Studieren hilft"](https://www.ntz.de/nuertingen/artikel_hfwu-in-nuertingen-wie-kuenstliche-intelligenz-beim-studieren-hilft.html)**  
Feature article on how AI supports university studies, highlighting the collaborative research between Tobias Leiblein and Prof. Dr. Mathias Engel.

**📰 [Stuttgarter Zeitung - "Wie künstliche Intelligenz beim Lernen hilft"](https://www.stuttgarter-zeitung.de/inhalt.wissenschaftler-aus-nuertingen-wie-kuenstliche-intelligenz-beim-lernen-hilft.016cc0c8-debb-46b5-9fb4-8e99815dfcdb.html)**  
Article discussing how artificial intelligence assists in learning processes.

---

**Academic Impact:**  
These media appearances reflect the growing recognition of OpenTuneWeaver's innovative approach to democratizing AI dataset generation for educational institutions and the broader implications of semantic chunking technology.

**Press Contact:**  
For additional interviews or press inquiries: [mathias@opentuneweaver.com](mailto:mathias@opentuneweaver.com)

## 💖 Sponsorship & Support

If OpenTuneWeaver provides immense value in building out datasets over traditional €5,000-€10,000 consulting options, help us keep it thriving! We rely on voluntary contributions.

### 🎯 Community Support

**Perfect for individuals, students, and organizations who want to support our mission:**

- **☕ Coffee for ProfEngel - $5**: Fuel late-night coding sessions for ProfEngel! 
- **☕ Coffee for the Team - $30**: Fuel late-night coding sessions for contributors
- **💻 GPU Hour Sponsor - $110+**: Help us test models faster on rigorous instances

**Ready to support democratized AI development?**  
[![Sponsor](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=ff69b4)](https://github.com/sponsors/ProfEngel)

## License 📜

This project is licensed under the **Apache License 2.0**.

**Usage Rights:**
- ✅ **Private Use**: Completely free
- ✅ **Educational Institutions**: Free for research and teaching
- ✅ **Commercial Use**: Free

If deploying commercially, we politely ask you to consider the [Github Sponsor](https://github.com/sponsors/ProfEngel) link to acknowledge the value provided!

Full license terms can be found in the LICENSE file.

## Acknowledgments 🙏

OpenTuneWeaver would not be possible without these excellent open-source frameworks:

**Core Frameworks:**
- Gradio - Elegant, no-code web interface
- Ollama - Agile local LLM interfacing 
- Docling (IBM) - Best-in-class PDF processing
- Marker - Robust PDF-to-Markdown conversion

**Thanks to the entire open-source community!** 🎉

***

## Citation & Research 📚

If you use OpenTuneWeaver in your research, please cite our paper:

```bibtex
@article{opentuneweaver2025,
title={OpenTuneWeaver: Semantically-structured, Curatable LLM Fine-tuning Pipeline for Research and Education},
author={Engel, Prof. Dr. Mathias},
journal={arXiv preprint},
year={2024},
institution={Hochschule für Wirtschaft und Umwelt Nürtingen-Geislingen},
note={Funded by MWK Baden-Württemberg and Stifterverband Deutschland}
}
```

***

## Support 💬

Do you have questions, suggestions, or need support?

- 🐛 **Issues**: [GitHub Issues](https://github.com/ProfEngel/OpenTuneWeaver/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ProfEngel/OpenTuneWeaver/discussions)  
- 🎓 **Academic Collaboration**: [research@opentuneweaver.com](mailto:research@opentuneweaver.com)

***

**Created by Prof. Dr. Mathias Engel 2023-2025** - Let's make OpenTuneWeaver even greater together! 💪

_Made with ❤️ in Stuttgart / Nürtingen, Germany_

***

## About

Semantically-structured, curatable all-in-one LLM text pipeline

<div align="left">
  <img src="assets/otw_logo.png" alt="OpenTuneWeaver Logo" width="100">
</div>

**Prof. Dr. Mathias Engel - ProfEngel** 

<div align="left">
  <img src="assets/hfwu_logo_w.png" alt="Nürtingen-Geislingen University" width="100">
</div>
**Hochschule für Wirtschaft und Umwelt Nürtingen-Geislingen**  
*part-funded by MWK Baden-Württemberg and Stifterverband Deutschland*

## 🤝 Open for Contributions

Contributions are welcome!  
If you have ideas, improvements, or bug reports, feel free to open an **Issue** or submit a **Pull Request**.

## Star History
<a href="https://star-history.com/#ProfEngel/OpenTuneWeaver&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ProfEngel/OpenTuneWeaver&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ProfEngel/OpenTuneWeaver&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ProfEngel/OpenTuneWeaver&type=Date" />
  </picture>
</a>

### Topics

`llm` `dataset-generation` `ai` `machine-learning` `nlp` `semantic-chunking` `pdf-processing` `qa-generation` `benchmarking` `gradio` `educational-ai` `research-tools`
