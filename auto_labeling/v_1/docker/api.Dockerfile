FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

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
        ultralytics \
        python-multipart \
        requests \
        pyyaml

COPY auto_labeling /workspace/auto_labeling
COPY auto_labeling/v_1/docker/entrypoints/api.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8010
ENTRYPOINT ["/entrypoint.sh"]
