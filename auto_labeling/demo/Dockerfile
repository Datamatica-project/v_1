FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pyyaml \
    python-multipart

COPY . /workspace/auto_labeling/demo

RUN mkdir -p /workspace/auto_labeling/demo/data

EXPOSE 8014

CMD ["uvicorn", "auto_labeling.demo.server:app", "--host", "0.0.0.0", "--port", "8014"]
