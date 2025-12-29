FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install \
        fastapi \
        uvicorn[standard] \
        pydantic \
        python-multipart \
        requests \
        pyyaml \
        ultralytics

# ❌ 코드 COPY 하지 않음
#   ./auto_labeling 은 docker-compose에서 mount

EXPOSE 8010

# 기본 실행 (compose에서 --reload 등으로 덮어쓰기 가능)
CMD ["uvicorn", "auto_labeling.v_1.api.main:create_app", "--host", "0.0.0.0", "--port", "8010"]
