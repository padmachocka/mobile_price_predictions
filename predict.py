import pickle
import pandas as pd

# Load the trained model
with open("best_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    preds = model.predict(df)
    return int(preds[0])

if __name__ == "__main__":
    sample = {
        "battery_power": 1043, "blue": 1, "clock_speed": 1.8, "dual_sim": 0,
        "fc": 14, "four_g": 1, "int_memory": 50, "m_dep": 0.9, "mobile_wt": 200,
        "n_cores": 8, "pc": 13, "px_height": 746, "px_width": 1412, "ram": 2631,
        "sc_h": 19, "sc_w": 3, "talk_time": 10, "three_g": 1, "touch_screen": 1,
        "wifi": 1, "screen_area": 57, "battery_per_weight": 5.2, "ppi": 550
    }
    print("Prediction:", predict(sample))
