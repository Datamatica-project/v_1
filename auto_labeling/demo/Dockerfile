FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    curl git \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pyyaml \
    python-multipart \
    pillow \
    opencv-python-headless \
    ultralytics

COPY . /workspace

RUN mkdir -p /workspace/auto_labeling/demo/data

EXPOSE 8014

CMD ["uvicorn", "auto_labeling.demo.server:app", "--host", "0.0.0.0", "--port", "8014"]
