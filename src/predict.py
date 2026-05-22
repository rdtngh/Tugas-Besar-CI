"""
Modul prediksi untuk sistem prediksi stres mahasiswa.

Menggunakan model Neuro-Fuzzy / ANFIS sebagai metode utama,
serta fallback ke Fuzzy Inference System (FIS) jika ANFIS belum tersedia.
"""

from pathlib import Path
from src.anfis_model import load_anfis_model, load_scaler, predict_anfis
from src.fuzzy_model import fuzzy_predict

ANFIS_MODEL_PATH = Path("models/anfis_model.pkl")
ANFIS_SCALER_PATH = Path("models/anfis_scaler.pkl")
_cached_anfis = {
    "model": None,
    "scaler": None,
}


def load_anfis():
    """Muat ANFIS model dan scaler jika tersedia."""
    if _cached_anfis["model"] is not None and _cached_anfis["scaler"] is not None:
        return _cached_anfis["model"], _cached_anfis["scaler"]

    if not ANFIS_MODEL_PATH.exists() or not ANFIS_SCALER_PATH.exists():
        raise FileNotFoundError("ANFIS model atau scaler tidak ditemukan.")

    model = load_anfis_model(ANFIS_MODEL_PATH)
    scaler = load_scaler(ANFIS_SCALER_PATH)
    _cached_anfis["model"] = model
    _cached_anfis["scaler"] = scaler
    return model, scaler


def predict_stress(input_data: dict):
    """Prediksi stres menggunakan ANFIS jika tersedia, else fallback FIS."""
    try:
        anfis_model, scaler = load_anfis()
        result = predict_anfis(input_data, anfis_model, scaler)
        return result["label"], result["score"], result["class_scores"]

    except Exception as e:
        print("ANFIS fallback: {0}".format(e))
        fuzzy_result = fuzzy_predict(input_data)
        return fuzzy_result["label"], fuzzy_result["score"], fuzzy_result["class_scores"]
