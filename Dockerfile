FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

WORKDIR /app

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

# Install production Python dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless

# Download official InsightFace buffalo_l models directly from GitHub releases
RUN mkdir -p /root/.insightface/models/buffalo_l && \
    curl -L https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -o /tmp/buffalo_l.zip && \
    unzip -q -o /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm /tmp/buffalo_l.zip

# Copy source code, model weights, and frontend assets
COPY app/ ./app/
COPY models/ ./models/
COPY static/ ./static/
COPY reports/ ./reports/

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
