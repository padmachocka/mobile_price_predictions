# mobile_price_predictions

FastAPI service + training pipeline for predicting mobile phone prices.

## Project Layout
- `app.py` – FastAPI app (serving predictions)
- `predict.py` – CLI/helper for running predictions
- `pipeline.py` – Training pipeline
- `mobile_price_prediction.py` – Notebook/script for analysis
- `requirements.txt` – Python deps
- `best_xgb_model.pkl` – Trained model (tracked with Git LFS recommended)
- `train.csv` – Sample training data (LFS recommended)
- `Dockerfile` – Container build

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
# health check
curl http://localhost:8000/health
