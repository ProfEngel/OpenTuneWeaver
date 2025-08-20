# 🐳 OpenTuneWeaver Docker Setup

## Universal API Support Edition

This Docker setup allows you to use OpenTuneWeaver with any OpenAI-compatible API endpoint without installing Ollama locally.

> **📌 Note**: Docker images are automatically built and published via GitHub Actions when new releases are tagged.

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

1. **Create a `.env` file** with your API configuration:
```bash
# For existing Ollama instance
OPENAI_API_BASE=http://your-ollama-host:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=gemma3:12b-it-qat

# For vLLM
# OPENAI_API_BASE=http://your-vllm-host:8000/v1
# OPENAI_API_KEY=token-abc123
# OPENAI_MODEL=meta-llama/Llama-3-8B

# For OpenAI
# OPENAI_API_BASE=https://api.openai.com/v1
# OPENAI_API_KEY=sk-your-key-here
# OPENAI_MODEL=gpt-4

# Optional: HuggingFace tokens
HF_TOKEN=hf_your_token_here
HF_WRITE_TOKEN=hf_your_write_token_here
```

2. **Start the container**:
```bash
docker-compose up -d
```

3. **Access OpenTuneWeaver**:
```
http://localhost:8080
```

### Option 2: Docker Run

```bash
docker run -d \
  --name opentuneweaver \
  -p 8080:8080 \
  -e OPENAI_API_BASE=http://your-api:11434/v1 \
  -e OPENAI_API_KEY=ollama \
  -e OPENAI_MODEL=gemma3:12b-it-qat \
  -v $(pwd)/data:/app/OpenTuneWeaver/pipeline/data \
  -v $(pwd)/models:/app/OpenTuneWeaver/pipeline/modules/06_finetuning/CustomModel \
  --gpus all \
  profengel/opentuneweaver:universal
```

### Option 3: Build from Source

```bash
# Clone repository
git clone https://github.com/ProfEngel/OpenTuneWeaver.git
cd OpenTuneWeaver

# Build Docker image
docker build -t opentuneweaver:local .

# Run with docker-compose
docker-compose up -d
```

## 📦 Available Docker Images

### Docker Hub
- `profengel/opentuneweaver:latest` - Latest stable version
- `profengel/opentuneweaver:universal` - Universal API support
- `profengel/opentuneweaver:cuda` - With CUDA support
- `profengel/opentuneweaver:v1.0.0` - Specific version

### GitHub Container Registry
- `ghcr.io/profengel/opentuneweaver:latest`
- `ghcr.io/profengel/opentuneweaver:universal`
- `ghcr.io/profengel/opentuneweaver:cuda`

## 🔧 Configuration Examples

### Using Local Ollama
```yaml
environment:
  - OPENAI_API_BASE=http://host.docker.internal:11434/v1
  - OPENAI_API_KEY=ollama
  - OPENAI_MODEL=llama3.2:3b
```

### Using Remote Ollama
```yaml
environment:
  - OPENAI_API_BASE=http://192.168.1.100:11434/v1
  - OPENAI_API_KEY=ollama
  - OPENAI_MODEL=gemma3:12b-it-qat
```

### Using vLLM
```yaml
environment:
  - OPENAI_API_BASE=http://vllm-server:8000/v1
  - OPENAI_API_KEY=token-abc123
  - OPENAI_MODEL=mistralai/Mistral-7B-Instruct-v0.2
```

### Using LocalAI
```yaml
environment:
  - OPENAI_API_BASE=http://localai:8080/v1
  - OPENAI_API_KEY=sk-localai
  - OPENAI_MODEL=ggml-model
```

### Using OpenAI API
```yaml
environment:
  - OPENAI_API_BASE=https://api.openai.com/v1
  - OPENAI_API_KEY=sk-your-openai-key
  - OPENAI_MODEL=gpt-4
```

## 🗂️ Volume Mounts

The following volumes are recommended for persistent data:

```yaml
volumes:
  - ./data:/app/OpenTuneWeaver/pipeline/data           # Training data
  - ./models:/app/OpenTuneWeaver/pipeline/modules/06_finetuning/CustomModel  # Trained models
  - ./logs:/app/OpenTuneWeaver/logs                    # Application logs
  - ./cache:/app/OpenTuneWeaver/cache                  # Cache directory
  - ./documents:/app/OpenTuneWeaver/pipeline/modules/01_convert/UPLOAD  # Upload directory
```

## 🖥️ GPU Support

### NVIDIA GPU
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Or with docker run:
```bash
docker run --gpus all ...
```

### Requirements:
- NVIDIA Docker runtime installed
- CUDA 12.1+ compatible GPU
- NVIDIA drivers 525.60.13+

## 🔍 Debugging

### View Logs
```bash
# Docker Compose
docker-compose logs -f opentuneweaver

# Docker
docker logs -f opentuneweaver
```

### Access Container Shell
```bash
docker exec -it opentuneweaver bash
```

### Test API Connection
```bash
# From inside container
curl http://your-api:11434/api/tags

# Test OpenAI-compatible endpoint
curl http://your-api:11434/v1/models \
  -H "Authorization: Bearer your-key"
```

## 🌐 Network Configuration

### Docker Networks
The compose file creates a bridge network `otw-network` for service communication.

### Host Networking
To access services on the host machine from within Docker:
- Linux: Use actual IP address (e.g., `192.168.1.100`)
- Mac/Windows: Use `host.docker.internal`

## 🔐 Security Considerations

1. **API Keys**: Store sensitive API keys in `.env` file (never commit to git)
2. **Network**: Use internal Docker networks for service communication
3. **Volumes**: Set appropriate permissions on mounted volumes
4. **Ports**: Only expose necessary ports

## 📊 Resource Limits

Set resource limits in docker-compose.yml:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 16G
    reservations:
      cpus: '2'
      memory: 8G
```

## 🔄 Updating

### Pull Latest Image
```bash
docker-compose pull
docker-compose up -d
```

### Rebuild After Changes
```bash
docker-compose build --no-cache
docker-compose up -d
```

## 🛠️ Troubleshooting

### Container Won't Start
- Check logs: `docker logs opentuneweaver`
- Verify API endpoint is reachable
- Ensure ports aren't already in use

### GPU Not Detected
- Verify NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
- Check CUDA installation: `nvidia-smi`

### API Connection Failed
- Test from host: `curl http://your-api:11434/api/tags`
- Check firewall rules
- Verify API service is running

### Out of Memory
- Reduce batch size in configuration
- Use 4-bit quantization
- Allocate more memory to Docker

## 📋 Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_BASE` | API endpoint URL | - | Yes |
| `OPENAI_API_KEY` | API authentication key | ollama | Yes |
| `OPENAI_MODEL` | Model name to use | - | Yes |
| `HF_TOKEN` | HuggingFace read token | - | No |
| `HF_WRITE_TOKEN` | HuggingFace write token | - | No |
| `GRADIO_SERVER_PORT` | Web UI port | 8080 | No |
| `CUDA_VISIBLE_DEVICES` | GPU device IDs | 0 | No |

## 🚢 Deployment Options

### Production Deployment
```yaml
services:
  opentuneweaver:
    image: profengel/opentuneweaver:universal
    restart: always
    deploy:
      replicas: 1
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
```

### Development Setup
```yaml
services:
  opentuneweaver:
    build: .
    volumes:
      - ./:/app/OpenTuneWeaver  # Mount source code
    environment:
      - DEBUG=true
```

## 📚 Additional Resources

- [Main README](https://github.com/ProfEngel/OpenTuneWeaver/blob/main/README.md)
- [Setup without Docker](https://github.com/ProfEngel/OpenTuneWeaver/blob/main/setup_universal.sh)
- [API Documentation](https://github.com/ProfEngel/OpenTuneWeaver/wiki/API-Configuration)
- [Troubleshooting Guide](https://github.com/ProfEngel/OpenTuneWeaver/wiki/Troubleshooting)

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/ProfEngel/OpenTuneWeaver/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ProfEngel/OpenTuneWeaver/discussions)
- **Email**: [support@opentuneweaver.com](mailto:support@opentuneweaver.com)

---

**Created by Prof. Dr. Mathias Engel** - Making LLM fine-tuning accessible to everyone! 🚀