FROM docker.io/pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    vim \
    wget \
    git \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core dependencies
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib \
    tqdm \
    jiwer

# Install audio processing
RUN pip install --no-cache-dir \
    librosa \
    soundfile

# Install wandb (before NeMo)
RUN pip install --no-cache-dir "wandb<0.19"

# Install pyarrow with pre-built wheel
RUN pip install --no-cache-dir --only-binary=:all: pyarrow

# Install NeMo toolkit for ASR
RUN pip install --no-cache-dir nemo_toolkit[asr]

WORKDIR /workspace

# Set environment
ENV PYTHONPATH=/workspace:$PYTHONPATH