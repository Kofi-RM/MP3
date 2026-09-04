FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV's Linux wheel needs these shared libraries even without a GUI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt && python -m pip check

COPY scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py

# Explicit copies keep old uploads, local databases, and secrets out of the image.
COPY app.py gunicorn.conf.py yolov8n.pt ./
COPY templates/ templates/
COPY static/css/styles.css static/css/compare.css static/css/yolo.css static/css/
COPY static/js/compare.js static/js/yolo.js static/js/
COPY static/images/car1.jpg static/images/car1.jpg

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/.runtime \
    && chown -R appuser:appuser /app/.runtime /app/.models
USER appuser

ENV APP_ENV=production \
    VIT_MODEL_PATH=/app/.models/vit \
    HF_HUB_OFFLINE=1 \
    YOLO_AUTOINSTALL=false \
    YOLO_OFFLINE=true \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

EXPOSE 10000
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
