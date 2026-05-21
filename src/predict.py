import joblib
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/stress_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")
LABEL_MAP = {
    0: "Rendah",
    1: "Sedang",
    2: "Tinggi",
}


def fallback_predict(input_data: dict):
    """Prediksi sederhana ketika model belum tersedia."""
    anxiety = input_data["anxiety_level"] / 20.0
    sleep = 1.0 - (input_data["sleep_quality"] / 5.0)
    study = input_data["study_load"] / 5.0
    performance = 1.0 - (input_data["academic_performance"] / 5.0)
    support = 1.0 - (input_data["social_support"] / 5.0)
    career = input_data["future_career_concerns"] / 5.0

    score = (
        anxiety * 0.40
        + sleep * 0.18
        + study * 0.15
        + performance * 0.15
        + support * 0.08
        + career * 0.04
    )

    if score < 0.33:
        label = "Rendah"
    elif score < 0.66:
        label = "Sedang"
    else:
        label = "Tinggi"

    return label, score


def predict_stress(input_data: dict):
    """Melakukan prediksi menggunakan model jika tersedia, atau fallback jika tidak."""
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            feature_order = [
                "anxiety_level",
                "sleep_quality",
                "study_load",
                "academic_performance",
                "social_support",
                "future_career_concerns",
            ]
            values_df = pd.DataFrame(
                [[input_data[f] for f in feature_order]],
                columns=feature_order,
            )
            values_scaled = scaler.transform(values_df)
            values_scaled_df = pd.DataFrame(values_scaled, columns=feature_order)
            prediction = model.predict(values_scaled_df)
            label = LABEL_MAP.get(int(prediction[0]), "Sedang")
            score = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(values_scaled_df)[0]
                score = float(np.max(proba))
            return label, score
        except Exception:
            return fallback_predict(input_data)
    return fallback_predict(input_data)
