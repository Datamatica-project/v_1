FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

RUN pip install \
    fastapi \
    uvicorn[standard] \
    ultralytics \
    pyyaml \
    numpy

COPY app.py /app/app.py
COPY test.py /app/test.py
COPY yolov11x.py /app/yolov11x.py
RUN mkdir -p /app/models
COPY models/student_v1.pt /app/models/student_v1.pt
ENV DEFAULT_WEIGHTS_PATH=/app/models/student_v1.pt

EXPOSE 8013
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8013"]
