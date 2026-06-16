FROM python:3.11-slim-bookworm

WORKDIR /usr/app/yaclip

RUN apt-get -y update --allow-releaseinfo-change \
    && apt-get -y install --no-install-recommends \
        libegl1 \
        libgles2 \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        git \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY requirements.txt .

RUN uv venv \
    && uv pip install --no-cache \
        pip \
        setuptools \
        wheel \ 
    && uv pip install --no-cache -r requirements.txt \
    && uv pip install --no-cache --force-reinstall --no-deps opencv-python-headless

COPY . .

EXPOSE 7860

ENTRYPOINT ["uv", "run", "app.py", "serve"]
