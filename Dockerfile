FROM python:3.10-slim

# Fast runtime + set model path for the app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/best_xgb_model.pkl

WORKDIR /app

# XGBoost runtime dep
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model
# (explicit copy avoids pulling junk into the image)
COPY app.py .
COPY best_xgb_model.pkl .

EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
