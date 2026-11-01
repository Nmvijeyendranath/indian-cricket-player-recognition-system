FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    HOME=/home/user

# Install Linux system libraries needed by OpenCV and InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (UID 1000)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install production Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless

# Download InsightFace buffalo_l models and clean unused 3D models to save disk & RAM
RUN mkdir -p /home/user/.insightface/models/buffalo_l && \
    curl -L https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -o /tmp/buffalo_l.zip && \
    unzip -q -o /tmp/buffalo_l.zip -d /home/user/.insightface/models/buffalo_l && \
    rm -f /tmp/buffalo_l.zip && \
    rm -f /home/user/.insightface/models/buffalo_l/1k3d68.onnx && \
    rm -f /home/user/.insightface/models/buffalo_l/2d106det.onnx && \
    rm -f /home/user/.insightface/models/buffalo_l/genderage.onnx && \
    chown -R user:user /home/user/.insightface

# Copy application files, ultra-compact model (606 KB), and frontend assets
COPY --chown=user:user app/ ./app/
COPY --chown=user:user models/ ./models/
COPY --chown=user:user static/ ./static/
COPY --chown=user:user reports/ ./reports/

USER user
EXPOSE 10000 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
