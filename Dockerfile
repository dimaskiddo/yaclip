FROM python:3.11-slim-bookworm
MAINTAINER Dimas Restu Hidayanto <drh.dimasrestu@gmail.com>

LABEL maintainer="Dimas Restu Hidayanto <drh.dimasrestu@gmail.com>"

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Jakarta \
    HOME=/

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/app

# Install system deps: EGL/GLES/GL for MediaPipe, FFmpeg for video processing.
RUN apt-get -y update --allow-releaseinfo-change \
    && apt-get -y dist-upgrade \
    && apt-get -y install --no-install-recommends \
        libegl1 \
        libgles2 \
        libgl1 \
        ffmpeg \
    && apt-get -y purge --autoremove \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager).
RUN pip3 install --no-cache-dir --upgrade --break-system-packages \
      uv

# Copy dependency manifest first (Docker layer caching).
COPY requirements.txt .

# Create venv and install packages.
# GPU torch is NOT installed — CPU-only torch is the safe default for Docker.
# opencv-python-headless prevents WSL/headless display crashes.
RUN uv venv \
    && uv pip install --no-cache --upgrade \
        pip \
        setuptools \
        wheel \
    && uv pip install --no-cache -r requirements.txt \
    && uv pip install --no-cache --force-reinstall --no-deps \
        opencv-python-headless

# Copy the rest of the application.
COPY . .

# Expose Gradio WebUI port.
EXPOSE 7860

# Default: serve the WebUI (override via CLI args: `clip <url>`, `cache purge`, etc.).
# The workspace auto-initialises on first boot (FFmpeg, fonts, dirs).
ENTRYPOINT ["uv", "run", "app.py"]
CMD ["serve"]
