# Dockerfile
FROM python:3.10-slim

# Fast runtime; point app to the full pipeline artifact
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/best_xgb_pipeline.pkl \
    PORT=8000

WORKDIR /app

# Native runtime deps (XGBoost) + curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (leverages layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and the pipeline artifact
COPY app.py .
COPY best_xgb_pipeline.pkl .

# (Optional) non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
