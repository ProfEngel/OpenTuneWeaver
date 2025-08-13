# OpenTuneWeaver 🧬

![GitHub stars](https://img.shields.io/github/stars/ProfEngel/OpenTuneWeaver?style=social)
![GitHub forks](https://img.shields.io/github/forks/ProfEngel/OpenTuneWeaver?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/ProfEngel/OpenTuneWeaver?style=social)
![GitHub repo size](https://img.shields.io/github/repo-size/ProfEngel/OpenTuneWeaver)
![GitHub language count](https://img.shields.io/github/languages/count/ProfEngel/OpenTuneWeaver)
![GitHub top language](https://img.shields.io/github/languages/top/ProfEngel/OpenTuneWeaver)
![GitHub last commit](https://img.shields.io/github/last-commit/ProfEngel/OpenTuneWeaver?color=red)
[![YouTube](https://img.shields.io/badge/YouTube-MatMaxEngel-red?logo=youtube&logoColor=white)](https://www.youtube.com/user/MatMaxEngel)
[![Website](https://img.shields.io/badge/Website-opentuneweaver.com-blue?logo=googlechrome&logoColor=white)](https://opentuneweaver.com/)

<div align="center">
  <img src="assets/otw_logo_banner.png" alt="OpenTuneWeaver Logo" width="900">
</div>

 



![OpenTuneWeaver Demo](assets/OTW_demo.gif)

**OpenTuneWeaver is a semantically-structured, curatable all-in-one LLM fine-tuning pipeline that automatically creates structured wiki entries, InstructQA datasets, and benchmarkable, deployment-ready models from any raw data (PDF, DOCX, etc.).** The system revolutionizes LLM fine-tuning through **semantic chunking**, **curatable dataset creation**, and **end-to-end automation** without requiring technical expertise.

<div align="right">
  <img src="assets/mwk_logo_w2.png" alt="Ministry of Science, Research and Arts Logo" height="60">
  <img src="assets/stifterverband_logo.jpg" alt="Stifterverband Logo" height="60">
</div>

This project is part-funded by the **Ministry of Science, Research and Arts Baden-Württemberg (MWK)** and **Stifterverband Deutschland** as part of digital [Fellowship-Program](https://www.stifterverband.org/bwdigifellows/2024_engel_leiblein).

> **Tip**  
> **Looking for an [Enterprise Plan](mailto:sales@opentuneweaver.com)?** – **Speak with Our Sales Team Today!**  
> Get **enhanced capabilities**, including **custom theming and branding**, **Service Level Agreement (SLA) support**, **Long-Term Support (LTS) versions**, **priority support**, **on-premise deployment**, and **more!**

***

![OpenTuneWeaver Viewer Demo](assets/OTW_viewer_demo.gif)
With the OTW-Viewer, all generated documents (converted Markdown files, lexicon wiki entries, QA instruct datasets, benchmark question datasets) can be read and curated as well as edited and saved back. Additionally, reports about the benchmark run and the pipeline run can be displayed.

## Key Features 🚀

- 🔄 **End-to-End Automation**: Only platform from PDF to deployment-ready, benchmarkable model in one workflow
- 🧠 **Semantic Wiki Chunking**: Revolutionary meaning-preserving segmentation instead of destructive fixed chunking  
- 📚 **Automatic Dataset Creation**: Wiki entries, InstructQA with 5 question types, benchmarks with ground truth
- 🎨 **Curatable Viewer Environment**: Interactive quality assurance with split/merge/annotation for all pipeline steps
- 📊 **Integrated Telemetry**: Real-time monitoring, metrics and audit trails for complete transparency
- 🤖 **GPU-Adaptive Training**: Automatic hardware optimization with LoRA/QLoRA for 100+ models
- 📱 **No-Code Gradio Interface**: Drag-&-drop upload with live terminal and complete pipeline control
- 🌐 **Multi-Format Export**: LoRA, Merged (both for transformers, vLLM, etc.), GGUF in Q_8 with quantizations for local deployment (OpenWebUI/LM-Studio)
- 🔍 **VLM Integration**: Vision-Language-Models for automatic image descriptions in documents
- ⚡ **Universal API Support**: Works with OpenAI, OpenRouter, Ollama, LM Studio, and any OpenAI-compatible API

***

## How to Install 🚀

### System Requirements

**Hardware:**
- **Linux system recommended** (Ubuntu 22.04 LTS or similar)
- **At least 100 GB free storage space**
- **For Training: NVIDIA GPU with at least 20 GB VRAM** (depending on the model being trained)
  - RTX 4090/A6000/A100 recommended
  - For smaller models: RTX 3090/4080 (16GB) possible
- **For Dataset Generation Only: No GPU required** (can use cloud APIs)
- **CUDA 12.8+ and cuDNN** (only if using local GPU)

**Accounts:**
- **HuggingFace Account** with Access Token (Read + optional Write)
- **API Access** (choose one):
  - OpenAI API Key
  - OpenRouter API Key
  - Ollama (local installation)
  - LM Studio (local installation)
  - Any OpenAI-compatible API endpoint

### HuggingFace Token Setup

1. Create an account on [huggingface.co](https://huggingface.co)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token with **Read** permission (and **Write** for model upload)
4. Note down the token for installation

### Universal Installation (NEW - Works with any API)

OpenTuneWeaver now supports **any OpenAI-compatible API** for dataset generation. Choose your preferred installation method:

#### Quick Installation with Direct Script

```bash
# Download and run the universal setup script
wget https://raw.githubusercontent.com/ProfEngel/OpenTuneWeaver/main/setup_universal.sh
chmod +x setup_universal.sh

# Configure your API (choose one):

# Option 1: For OpenAI
export OPENAI_API_TYPE=openai
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_API_KEY=sk-your-key-here
export OPENAI_MODEL_NAME=gpt-4

# Option 2: For OpenRouter
export OPENAI_API_TYPE=openrouter
export OPENAI_API_BASE=https://openrouter.ai/api/v1
export OPENAI_API_KEY=your-openrouter-key
export OPENAI_MODEL_NAME=meta-llama/llama-3.2-3b-instruct

# Option 3: For local Ollama (default)
export OPENAI_API_TYPE=ollama
export OPENAI_API_BASE=http://localhost:11434/v1
export OPENAI_MODEL_NAME=llama3.2:latest

# Option 4: For LM Studio
export OPENAI_API_TYPE=lmstudio
export OPENAI_API_BASE=http://localhost:1234/v1
export OPENAI_MODEL_NAME=your-loaded-model

# Run the installation
./setup_universal.sh
```

#### Installation with Virtual Environment (Recommended)

```bash
# Create and activate virtual environment
python3 -m venv opentuneweaver-env
source opentuneweaver-env/bin/activate

# Clone repository
git clone https://github.com/ProfEngel/OpenTuneWeaver.git
cd OpenTuneWeaver

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure API (see options above)
export OPENAI_API_TYPE=openai  # or your preferred API
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_API_KEY=your-api-key
export OPENAI_MODEL_NAME=gpt-4

# Run setup
./setup_universal.sh
```

#### Installation with Conda

```bash
# Create conda environment
conda create -n opentuneweaver python=3.11
conda activate opentuneweaver

# Clone repository
git clone https://github.com/ProfEngel/OpenTuneWeaver.git
cd OpenTuneWeaver

# Install dependencies
pip install -r requirements.txt

# Install unsloth (for training)
pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth-zoo.git

# Configure API (see options above)
export OPENAI_API_TYPE=your-api-type
export OPENAI_API_BASE=your-api-base-url
export OPENAI_API_KEY=your-api-key
export OPENAI_MODEL_NAME=your-model

# Run setup
./setup_universal.sh
```

#### Docker Installation (Recommended for Production)

```bash
# Clone repository
git clone https://github.com/ProfEngel/OpenTuneWeaver.git
cd OpenTuneWeaver

# Copy and configure environment
cp .env.example .env
# Edit .env with your API settings

# Build and run with Docker Compose
docker-compose up -d

# Access at http://localhost:8080
```

### Runpod Installation (For GPU Training)

**Runpod Template:**
```
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
Disk Volume: 100 GB
Pod Volume:  100 GB
Open Ports: 8080,11434
```

**Installation:**
```bash
cd /workspace
git clone https://github.com/ProfEngel/OpenTuneWeaver.git
cd OpenTuneWeaver

# For Runpod with Ollama (local inference)
./setup_runpod_direct.sh

# OR for Runpod with external API
export OPENAI_API_TYPE=openai
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_API_KEY=your-key
export OPENAI_MODEL_NAME=gpt-4
./setup_universal.sh
```

### API Configuration Examples

#### Using OpenAI GPT-4
```bash
export OPENAI_API_TYPE=openai
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...your-key...
export OPENAI_MODEL_NAME=gpt-4  # or gpt-3.5-turbo
```

#### Using OpenRouter
```bash
export OPENAI_API_TYPE=openrouter
export OPENAI_API_BASE=https://openrouter.ai/api/v1
export OPENAI_API_KEY=your-openrouter-key
export OPENAI_MODEL_NAME=meta-llama/llama-3.2-3b-instruct
# Other models: claude-3-opus, mistral-large, etc.
```

#### Using Local Ollama
```bash
# First install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# Configure OpenTuneWeaver
export OPENAI_API_TYPE=ollama
export OPENAI_API_BASE=http://localhost:11434/v1
export OPENAI_MODEL_NAME=llama3.2:latest
```

#### Using LM Studio
```bash
# Start LM Studio and load a model
# Then configure:
export OPENAI_API_TYPE=lmstudio
export OPENAI_API_BASE=http://localhost:1234/v1
export OPENAI_MODEL_NAME=your-loaded-model
```

#### Using Custom API Endpoint
```bash
export OPENAI_API_TYPE=custom
export OPENAI_API_BASE=https://your-api-endpoint.com/v1
export OPENAI_API_KEY=your-api-key
export OPENAI_MODEL_NAME=your-model-name
```

### Starting OpenTuneWeaver

After installation, start the application:

```bash
# Direct start
./start_otw.sh

# Or with custom port
export SERVER_PORT=7860
./start_otw.sh

# Access the UI
# Local: http://localhost:8080
# Remote: http://your-server-ip:8080
```

### Troubleshooting

If you encounter issues:

```bash
# Check installation
./debug_otw.sh

# View logs
tail -f logs/pipeline.log

# Test API connection
curl -X POST $OPENAI_API_BASE/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "'$OPENAI_MODEL_NAME'", "messages": [{"role": "user", "content": "Test"}]}'
```

***

## What's Next? 🌟

**Short to medium-term roadmap:**
- 🌐 **English Localization**: Complete UI and documentation in English
- 🌍 **Multilingual Support**: Spanish, French, additional languages
- 🤖 **Extended Model Support**: 
  - GPT-OSS family
  - Qwen 2.5/3.0 series  
  - Mixtral and other MoE models
  - Phi-4 and Microsoft models
- 📊 **Advanced Analytics Dashboard**: Detailed training metrics and comparisons
- 🔧 **API Interface**: RESTful API for external integration
- 📱 **Mobile-optimized UI**: Responsive design for tablets and smartphones

See our complete [Roadmap](docs/roadmap.md) for detailed information.

***

## License 📜

This project is licensed under the **Apache License 2.0**. 

**Usage Rights:**
- ✅ **Private Use**: Completely free
- ✅ **Educational Institutions**: Free for research and teaching
- ✅ **Open Source Projects**: Free use with attribution
- ⚠️ **Commercial Use**: Enterprise plan required

For commercial use, contact our [Sales Team](mailto:sales@opentuneweaver.com) for an Enterprise Plan.

Full license terms can be found in the [LICENSE](LICENSE) file.

***

## Acknowledgments 🙏

OpenTuneWeaver would not be possible without these excellent open-source frameworks:

**Core Frameworks:**
- [Unsloth](https://github.com/unslothai/unsloth) - Efficient LLM fine-tuning
- [Gradio](https://gradio.app/) - No-code web interface
- [Transformers](https://github.com/huggingface/transformers) - HuggingFace Model Hub
- [Ollama](https://ollama.ai/) - Local LLM inference

**Document Processing:**
- [Docling (IBM)](https://github.com/DS4SD/docling) - Intelligent PDF processing
- [Marker](https://github.com/VikParuchuri/marker) - PDF-to-Markdown conversion

**Training & Optimization:**
- [LoRA/QLoRA](https://github.com/microsoft/LoRA) - Parameter-efficient fine-tuning
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) - GPU quantization
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning

**Vision & Multimodal:**
- [Google Gemma](https://ai.google.dev/gemma) - Vision-Language models
- [OpenAI CLIP](https://github.com/openai/CLIP) - Image-text understanding

**Thanks to the entire open-source community!** 🎉

***

## Citation & Research 📚

If you use OpenTuneWeaver in your research, please cite our paper:

```bibtex
@article{opentuneweaver2024,
  title={OpenTuneWeaver: Semantically-structured, Curatable LLM Fine-tuning Pipeline for Research and Education},
  author={Engel, Prof. Dr. Mathias},
  journal={arXiv preprint},
  year={2024},
  institution={Hochschule für Wirtschaft und Umwelt Nürtingen-Geislingen},
  note={Funded by MWK Baden-Württemberg and Stifterverband Deutschland}
}
```

**Paper available:**
- 📄 [Local Version](docs/opentuneweaver-paper.pdf)
- 🌐 [ArXiv](https://arxiv.org/) *(Link follows after publication)*

***

## Support 💬

Do you have questions, suggestions, or need support?

- 🐛 **Issues**: [GitHub Issues](https://github.com/ProfEngel/OpenTuneWeaver/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ProfEngel/OpenTuneWeaver/discussions)  
- 📧 **Enterprise Support**: [sales@opentuneweaver.com](mailto:sales@opentuneweaver.com)
- 🎓 **Academic Collaboration**: [research@opentuneweaver.com](mailto:research@opentuneweaver.com)

***

**Created by Prof. Dr. Mathias Engel 2023-2025** - Let's make OpenTuneWeaver even greater together! 💪

***

## About

Semantically-structured, curatable all-in-one LLM fine-tuning pipeline

<div align="left">
  <img src="assets/otw_logo.png" alt="OpenTuneWeaver Logo" width="100">
</div>

**Prof. Dr. Mathias Engel - ProfEngel** 

<div align="left">
  <img src="assets/hfwu_logo_w.png" alt="Nürtingen-Geislingen University" width="100">
</div>
**Hochschule für Wirtschaft und Umwelt Nürtingen-Geislingen**  
*part-funded by MWK Baden-Württemberg and Stifterverband Deutschland*

## Star History
<a href="https://star-history.com/#ProfEngel/OpenTuneWeaver&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ProfEngel/OpenTuneWeaver&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ProfEngel/OpenTuneWeaver&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ProfEngel/OpenTuneWeaver&type=Date" />
  </picture>
</a>


### Topics

`llm` `finetuning` `ai` `machine-learning` `nlp` `semantic-chunking` `lora` `qlora` `pdf-processing` `qa-generation` `benchmarking` `gradio` `huggingface` `educational-ai` `research-tools`