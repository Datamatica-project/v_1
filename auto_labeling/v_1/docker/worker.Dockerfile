FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    git curl \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip

RUN python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch torchvision torchaudio

RUN python -m pip install \
    ultralytics \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pyyaml requests tqdm \
    opencv-python-headless

RUN python -c "import uvicorn, fastapi; print('uvicorn=', uvicorn.__version__)"

COPY auto_labeling /workspace/auto_labeling
COPY auto_labeling/v_1/docker/entrypoints/worker.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8011
ENTRYPOINT ["/entrypoint.sh"]
