FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# python -> python3 alias
RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip

# PyTorch CUDA 11.8
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch torchvision torchaudio

# worker runtime deps
RUN python -m pip install \
    ultralytics \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pyyaml \
    requests \
    tqdm \
    opencv-python-headless

# sanity check (optional)
RUN python - <<EOF
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
EOF

# ❌ 코드 COPY 하지 않음
#   ./auto_labeling 은 docker-compose에서 mount

EXPOSE 8011

# 기본 실행:
# - worker가 HTTP 서버면 그대로
# - loop runner면 compose에서 command로 덮어쓰기
CMD ["uvicorn", "auto_labeling.v_1.worker.main:app", "--host", "0.0.0.0", "--port", "8011"]
