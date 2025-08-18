from fastapi import FastAPI
from predict import predict

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_api(data: dict):
    result = predict(data)
    return {"price_range": result}
