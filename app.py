# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional, Any
import os
import pandas as pd
import joblib

app = FastAPI(title="Mobile Price Predictor")

# --- Base/raw feature names from the dataset ---
RAW_COLS = [
    "battery_power","blue","clock_speed","dual_sim","fc","four_g","int_memory",
    "m_dep","mobile_wt","n_cores","pc","px_height","px_width","ram",
    "sc_h","sc_w","talk_time","three_g","touch_screen","wifi"
]

# Engineered features (computed in the service to match training)
ENGINEERED_COLS = ["screen_area", "battery_per_weight", "ppi"]
MODEL_COLS = RAW_COLS + ENGINEERED_COLS

# Prefer the full pipeline artifact (preprocessing + model)
MODEL_PATH = os.getenv("MODEL_PATH", "/app/best_xgb_pipeline.pkl")
LEGACY_MODEL_PATH = "/app/best_xgb_model.pkl"  # bare model (not recommended)

def _load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")
    if os.path.exists(LEGACY_MODEL_PATH):
        raise RuntimeError(
            f"Found legacy model '{LEGACY_MODEL_PATH}' but not pipeline '{MODEL_PATH}'. "
            "Your training used scaling/engineered features. Export and use the full pipeline "
            "(preprocessing + model), e.g. 'best_xgb_pipeline.pkl', and set MODEL_PATH accordingly."
        )
    raise RuntimeError(
        f"No model found. Expected pipeline at '{MODEL_PATH}'. "
        "Set MODEL_PATH env var to your pipeline artifact."
    )

model = _load_model()

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

def _expected_feature_names(m) -> Optional[list]:
    """Return the model's expected feature names/order if available."""
    names = getattr(m, "feature_names_in_", None)
    if names is not None:
        return list(names)
    booster = getattr(m, "get_booster", lambda: None)()
    if booster is not None:
        try:
            return list(getattr(booster, "feature_names", None) or [])
        except Exception:
            pass
    return None

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # Defensive replacements to avoid divide-by-zero
    df["sc_w"] = df["sc_w"].replace(0, 0.1)
    df["mobile_wt"] = df["mobile_wt"].replace(0, 0.1)

    # Screen area (cm^2)
    df["screen_area"] = df["sc_h"] * df["sc_w"]

    # Battery per weight
    df["battery_per_weight"] = df["battery_power"] / df["mobile_wt"]

    # Proper PPI: pixel diagonal / screen diagonal (inches)
    pixel_diag = (df["px_height"]**2 + df["px_width"]**2) ** 0.5
    cm_diag = (df["sc_h"]**2 + df["sc_w"]**2) ** 0.5
    diag_inches = cm_diag / 2.54
    ppi = pixel_diag / diag_inches.replace({0: float("nan")})
    df["ppi"] = ppi.fillna(0.0)

    # Keep a consistent column order for downstream alignment
    return df[MODEL_COLS]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug/model-info")
def debug_model_info():
    exp = _expected_feature_names(model)
    classes = getattr(model, "classes_", None)
    params = getattr(model, "get_xgb_params", lambda: {})()
    return {
        "model_path": MODEL_PATH,
        "expected_features": exp,
        "model_cols_service": MODEL_COLS,
        "classes_": list(map(int, classes)) if classes is not None else None,
        "xgb_params": params
    }

@app.post("/debug/predict-proba")
def debug_predict_proba(items: Union[MobileFeatures, List[MobileFeatures]]):
    payload = [items.model_dump()] if isinstance(items, MobileFeatures) else [i.model_dump() for i in items]
    df = pd.DataFrame(payload)

    # Validate required raw columns
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail={"error": "Missing columns", "missing": missing})

    df = _add_engineered_features(df)

    # Align to model feature order if the model exposes it
    expected = _expected_feature_names(model)
    if expected:
        miss_for_model = [c for c in expected if c not in df.columns]
        if miss_for_model:
            raise HTTPException(status_code=500, detail={"error": "Feature mismatch", "missing": miss_for_model})
        df = df[expected]

    try:
        proba = model.predict_proba(df)
        preds = model.predict(df)
        return {
            "predicted_class": [int(x) for x in preds],
            "proba": proba.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict_proba error: {e}")

@app.post("/predict")
def predict(items: Union[MobileFeatures, List[MobileFeatures]]):
    payload = [items.model_dump()] if isinstance(items, MobileFeatures) else [i.model_dump() for i in items]
    df = pd.DataFrame(payload)

    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    try:
        df = _add_engineered_features(df)

        # Align to model order if available (helps when the model was trained with named columns)
        expected = _expected_feature_names(model)
        if expected is not None:
            # Logically, warn if any mismatch, but proceed only when no missing features.
            missing_for_model = [c for c in expected if c not in df.columns]
            if missing_for_model:
                raise HTTPException(status_code=500, detail={"error": "Feature mismatch", "missing": missing_for_model})
            df = df[expected]

        preds = model.predict(df)
        return {"predictions": [int(x) for x in preds]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.get("/")
def root():
    return {"ok": True, "service": "mobile-price"}
