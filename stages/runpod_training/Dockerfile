# RunPod Dockerfile for Individual Item Detection Training
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python \
    matplotlib \
    pandas \
    numpy \
    Pillow \
    pyyaml \
    tqdm \
    seaborn \
    tensorboard

# Copy training files
COPY . /workspace/

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "train_individual_items.py"]
