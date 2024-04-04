# Dockerfile
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    DISPLAY=:1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        cmake \
        zlib1g-dev \
        libjpeg-dev \
        xvfb \
        xorg-dev \
        libboost-all-dev \
        libsdl2-dev \
        swig \
        ffmpeg \
        unzip \
        libpq-dev \
        libvulkan-dev \
        libgtk2.0-dev \
        libgl1-mesa-glx \
        x11-xserver-utils \
        x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and TensorFlow
RUN pip install --no-cache-dir \
        tensorflow

# Install RLlib and dependencies
RUN pip install --no-cache-dir \
        ray[rllib]

RUN pip install wandb

# Set up non-root user
RUN groupadd -r jonas && useradd -r -g jonas -m -s /bin/bash jonas
USER jonas

# Set working directory
WORKDIR /workspace
