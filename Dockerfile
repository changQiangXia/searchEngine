# NexusMind Docker Image
# Supports CUDA for GPU acceleration

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="NexusMind Team"
LABEL description="Next-gen multimodal semantic search engine"
LABEL version="0.1.0"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libopenblas-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies
# Note: We install CPU versions first, then override with CUDA versions
RUN pip3 install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install CPU-only dependencies first
RUN pip3 install --no-cache-dir \
    transformers==4.35.0 \
    accelerate==0.24.0 \
    pillow==10.0.0 \
    numpy==1.24.0 \
    pydantic==2.0.0 \
    pydantic-settings==2.0.0 \
    typer==0.9.0 \
    rich==13.0.0 \
    pyyaml==6.0 \
    psutil==5.9.0 \
    tqdm==4.66.0 \
    platformdirs==4.0.0 \
    streamlit==1.28.0 \
    plotly==5.18.0 \
    scikit-learn==1.3.0 \
    opencv-python==4.8.0

# Install faiss-gpu separately (may require additional system deps)
RUN pip3 install --no-cache-dir faiss-gpu==1.7.4 || pip3 install --no-cache-dir faiss-cpu==1.7.4

# Copy application code
COPY src/ ./src/
COPY apps/ ./apps/
COPY config/ ./config/
COPY tools/ ./tools/
COPY README.md LICENSE ./

# Install the package
RUN pip3 install -e .

# Create directories for data and cache
RUN mkdir -p /app/data/workspaces /app/data/cache /app/logs

# Set environment variables
ENV NEXUS_WORKSPACE_DIR=/app/data/workspaces
ENV NEXUS_CACHE_DIR=/app/data/cache
ENV NEXUS_LOG_DIR=/app/logs
ENV PYTHONPATH=/app/src

# Expose ports
# 8501 for Streamlit Web UI
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import nexus_mind; print('OK')" || exit 1

# Default command
CMD ["nexus", "status"]

# Usage:
# Build: docker build -t searchengine:latest .
# Run CLI: docker run --gpus all -it searchengine:latest nexus --help
# Run Web: docker run --gpus all -p 8501:8501 searchengine:latest streamlit run apps/web/app.py