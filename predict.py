# predict.py
import os
import pickle
import math
import pandas as pd
from typing import Dict, Any, Optional

PIPELINE_PATH = os.getenv("MODEL_PATH", "best_xgb_pipeline.pkl")
LEGACY_MODEL_PATH = "best_xgb_model.pkl"  # for helpful error if only this exists

# ---- Required base inputs (original Kaggle columns) ----
BASE_FEATURES = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi"
]

def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Defensive replacements to avoid divide-by-zero
    df["sc_w"] = df["sc_w"].replace(0, 0.1)
    df["mobile_wt"] = df["mobile_wt"].replace(0, 0.1)

    # Engineered features used in training
    df["screen_area"] = df["sc_h"] * df["sc_w"]
    df["battery_per_weight"] = df["battery_power"] / df["mobile_wt"]

    # Proper PPI: pixels per inch = pixel diagonal / screen diagonal (inches)
    diag_cm = (df["sc_h"]**2 + df["sc_w"]**2) ** 0.5
    diag_in = diag_cm / 2.54
    pix_diag = (df["px_height"]**2 + df["px_width"]**2) ** 0.5
    ppi = pix_diag / diag_in.replace(0, pd.NA)
    df["ppi"] = ppi.fillna(ppi.median())

    return df

def _load_pipeline():
    if os.path.exists(PIPELINE_PATH):
        with open(PIPELINE_PATH, "rb") as f:
            return pickle.load(f)
    # Helpful error if user still has only the bare model
    if os.path.exists(LEGACY_MODEL_PATH):
        raise RuntimeError(
            f"Found '{LEGACY_MODEL_PATH}' but not '{PIPELINE_PATH}'. "
            "Your training used scaling/engineered features. Please save and use the full pipeline "
            "(preprocessing + model) as 'best_xgb_pipeline.pkl' to ensure correct predictions."
        )
    raise FileNotFoundError(
        f"Model artifact not found. Expected pipeline at '{PIPELINE_PATH}'. "
        "Set MODEL_PATH env var if you used a different filename."
    )

_model_pipeline = _load_pipeline()

def predict(input_data: Dict[str, Any], return_proba: bool = False) -> Any:
    """
    input_data: dict with the 20 base features (raw). Engineered fields, if present, are ignored.
    return_proba: if True, returns dict of class->probability; else returns predicted class (int).
    """
    # Build DataFrame with only base features (ignore any extra keys)
    missing = [c for c in BASE_FEATURES if c not in input_data]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}. "
                         f"Provide all: {BASE_FEATURES}")

    base_row = {k: input_data[k] for k in BASE_FEATURES}
    df = pd.DataFrame([base_row])
    df = _to_numeric(df)

    # Engineer the same features as training
    df = _engineer_features(df)

    # Predict using the pipeline (handles scaling + model)
    if return_proba:
        proba = _model_pipeline.predict_proba(df)[0]
        # Classes assumed 0..3; adjust if your model classes differ
        classes = getattr(_model_pipeline, "classes_", [0,1,2,3])
        return {int(c): float(p) for c, p in zip(classes, proba)}
    else:
        pred = _model_pipeline.predict(df)[0]
        return int(pred)

if __name__ == "__main__":
    # Example with ONLY base features (engineered ones are computed internally)
    sample = {
        "battery_power": 1043, "blue": 1, "clock_speed": 1.8, "dual_sim": 0,
        "fc": 14, "four_g": 1, "int_memory": 50, "m_dep": 0.9, "mobile_wt": 200,
        "n_cores": 8, "pc": 13, "px_height": 746, "px_width": 1412, "ram": 2631,
        "sc_h": 19, "sc_w": 3, "talk_time": 10, "three_g": 1, "touch_screen": 1,
        "wifi": 1
    }
    print("Prediction:", predict(sample))
    print("Probabilities:", predict(sample, return_proba=True))
