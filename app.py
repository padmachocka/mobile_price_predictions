# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import os, math
import pandas as pd
import joblib

app = FastAPI(title="Mobile Price Predictor")

RAW_COLS = [
    "battery_power","blue","clock_speed","dual_sim","fc","four_g","int_memory",
    "m_dep","mobile_wt","n_cores","pc","px_height","px_width","ram",
    "sc_h","sc_w","talk_time","three_g","touch_screen","wifi"
]

# Model expects these columns IN THIS ORDER (note the last 3 engineered features)
MODEL_COLS = RAW_COLS + ["screen_area", "battery_per_weight", "ppi"]

MODEL_PATH = os.getenv("MODEL_PATH", "/app/lr_model.pkl")  # change if needed
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

class MobileFeatures(BaseModel):
    battery_power: int
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int

@app.get("/health")
def health():
    return {"status": "ok"}

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # screen area in cm^2
    df["screen_area"] = df["sc_h"] * df["sc_w"]

    # battery per weight (defend divide-by-zero)
    df["battery_per_weight"] = df["battery_power"] / df["mobile_wt"].replace({0: float("nan")})
    df["battery_per_weight"] = df["battery_per_weight"].fillna(0.0)

    # pixels per inch (PPI)
    # sc_h / sc_w are in cm -> diagonal inches = sqrt(sc_h^2 + sc_w^2) / 2.54
    # ppi = pixel_diagonal / diagonal_inches
    pixel_diag = (df["px_height"]**2 + df["px_width"]**2) ** 0.5
    cm_diag = (df["sc_h"]**2 + df["sc_w"]**2) ** 0.5
    diag_inches = cm_diag / 2.54
    # avoid division by zero
    df["ppi"] = pixel_diag / diag_inches.replace({0: float("nan")})
    df["ppi"] = df["ppi"].fillna(0.0)

    # ensure exact column order
    return df[MODEL_COLS]

@app.post("/predict")
def predict(items: Union[MobileFeatures, List[MobileFeatures]]):
    payload = [items.dict()] if isinstance(items, MobileFeatures) else [i.dict() for i in items]
    df = pd.DataFrame(payload)

    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    try:
        df = add_engineered_features(df)
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"predictions": [int(x) for x in preds]}
