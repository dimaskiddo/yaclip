FROM python:3.11-slim-bookworm
MAINTAINER Dimas Restu Hidayanto <drh.dimasrestu@gmail.com>

LABEL maintainer="Dimas Restu Hidayanto <drh.dimasrestu@gmail.com>"

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Jakarta \
    HOME=/

WORKDIR /usr/app/yaclip

RUN apt-get -y update --allow-releaseinfo-change \
    && apt-get -y dist-upgrade \
    && apt-get -y install --no-install-recommends \
        libegl1 \
        libgles2 \
        libgl1 \
        ffmpeg \
        git \
    && apt-get -y purge --autoremove \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages --upgrade \
        pip \
        setuptools \
        wheel \
        uv

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
