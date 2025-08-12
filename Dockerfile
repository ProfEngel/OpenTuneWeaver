# OpenTuneWeaver All-in-One Docker Image mit Ollama
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Setze Environment-Variablen
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV OLLAMA_HOST=http://localhost:11434

# System-Dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libreoffice \
    wkhtmltopdf \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-deu \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgconf-2-4 \
    fonts-liberation \
    supervisor \
    nginx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installiere Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python als python verfügbar machen
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Arbeitsverzeichnis
WORKDIR /workspace

# Kopiere requirements.txt
COPY requirements.txt .

# Installiere Python-Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Installiere Unsloth (speziell für Gemma)
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-deps "xformers<0.0.27" "trl<0.10.0" peft accelerate bitsandbytes

# Kopiere das Projekt
COPY . /workspace/OpenTuneWeaver/

# Wechsle ins Projektverzeichnis
WORKDIR /workspace/OpenTuneWeaver

# Baue llama.cpp im 06_finetuning Ordner
WORKDIR /workspace/OpenTuneWeaver/pipeline/modules/06_finetuning
RUN git clone --recursive https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release -j$(nproc) && \
    cd ../..

# Zurück zum Hauptverzeichnis
WORKDIR /workspace/OpenTuneWeaver

# Erstelle notwendige Verzeichnisse
RUN mkdir -p pipeline/data/OUTPUT \
    pipeline/modules/01_convert/UPLOAD \
    pipeline/modules/01_convert/INPUT \
    pipeline/modules/01_convert/OUTPUT \
    viewer/images \
    /root/.ollama \
    /var/log/supervisor

# Kopiere Supervisor-Konfiguration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Kopiere das angepasste Pipeline-Config mit localhost
COPY pipeline_config_local.json /workspace/OpenTuneWeaver/pipeline/pipeline_config.json

# Setze Permissions
RUN chmod -R 755 /workspace/OpenTuneWeaver

# Expose Ports
EXPOSE 8080 11434

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8080/ && curl -f http://localhost:11434/api/tags || exit 1

# Start-Script
COPY docker-entrypoint-ollama.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]