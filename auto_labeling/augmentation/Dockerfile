# Dockerfile
# FastAPI + Uvicorn
# Augmentation API (flip / scale / blur / color, cycle-based)

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OpenCV runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --upgrade pip && \
    pip install \
      fastapi==0.115.0 \
      uvicorn[standard]==0.30.6 \
      python-multipart==0.0.9 \
      numpy==1.26.4 \
      albumentations==2.0.1 \
      opencv-python==4.10.0.84

# Copy sources
COPY app.py ./app.py
COPY augmentation.py ./augmentation.py

# Expose port
EXPOSE 8012

# Run server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8012"]
