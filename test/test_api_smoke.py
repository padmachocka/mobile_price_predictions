# tests/test_api_smoke.py
import requests

BASE_URL = "http://18.175.251.68:8000"

low_spec_payload = {
    "battery_power": 775, "blue": 0, "clock_speed": 0.5, "dual_sim": 1,
    "fc": 0, "four_g": 0, "int_memory": 24, "m_dep": 0.8, "mobile_wt": 187,
    "n_cores": 1, "pc": 0, "px_height": 356, "px_width": 563, "ram": 373,
    "sc_h": 16, "sc_w": 3, "talk_time": 5, "three_g": 1, "touch_screen": 1,
    "wifi": 1
}

high_spec_payload = {
    "battery_power": 3000, "blue": 1, "clock_speed": 2.8, "dual_sim": 1,
    "fc": 24, "four_g": 1, "int_memory": 256, "m_dep": 0.2, "mobile_wt": 150,
    "n_cores": 8, "pc": 40, "px_height": 2200, "px_width": 3200, "ram": 8192,
    "sc_h": 20, "sc_w": 10, "talk_time": 24, "three_g": 1, "touch_screen": 1,
    "wifi": 1
}

def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_low_spec_prediction():
    resp = requests.post(f"{BASE_URL}/predict", json=low_spec_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    # Expect class 0 for low-end specs
    assert data["predictions"][0] == 0

def test_high_spec_prediction():
    resp = requests.post(f"{BASE_URL}/predict", json=high_spec_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    # Expect class 3 for high-end specs
    assert data["predictions"][0] == 3
