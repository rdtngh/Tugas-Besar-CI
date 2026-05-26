"""
Modul prediksi untuk sistem prediksi stres mahasiswa.

Alur prediksi:
1. Coba memuat dan gunakan model Simplified ANFIS (Neuro-Fuzzy) dari models/anfis_model.pkl
2. Jika ANFIS gagal (file tidak ditemukan atau error), fallback ke Fuzzy Inference System (FIS)
3. FIS menggunakan rule base IF-THEN yang didefinisikan secara statis

Output selalu berupa tuple: (label, score, class_scores, model_used)
model_used akan berisi string yang mengidentifikasi model mana yang digunakan untuk prediksi.
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
    """
    Prediksi tingkat stres mahasiswa.
    
    Alur:
    1. Coba gunakan model Simplified ANFIS (pendekatan Neuro-Fuzzy hybrid ANN+Fuzzy)
       - Model ini menggunakan membership function Gaussian adaptif yang dilatih dari data
       - Parameter model disimpan di models/anfis_model.pkl
    
    2. Jika ANFIS gagal dimuat atau error, gunakan Fuzzy Inference System (FIS) sebagai fallback
       - FIS menggunakan rule base IF-THEN manual dan membership function statis
       - FIS adalah baseline rule-based fuzzy inference yang sederhana
    
    Args:
        input_data: dict dengan kunci
            - anxiety_level (0-20)
            - sleep_quality (0-5)
            - study_load (0-5)
            - academic_performance (0-5)
            - social_support (0-5)
            - future_career_concerns (0-5)
    
    Returns:
        tuple: (label, score, class_scores, model_used)
            - label: 'Rendah', 'Sedang', atau 'Tinggi'
            - score: confidence score (0-1)
            - class_scores: dict dengan skor per kelas
            - model_used: string yang menunjukkan model mana yang digunakan
                "Simplified ANFIS" atau "Fuzzy Inference System (Fallback)"
    """
    try:
        # Coba muat dan gunakan model ANFIS
        anfis_model, scaler = load_anfis()
        result = predict_anfis(input_data, anfis_model, scaler)
        return result["label"], result["score"], result["class_scores"], "Simplified ANFIS"

    except FileNotFoundError as e:
        print(f"[WARNING] ANFIS model tidak ditemukan, fallback ke FIS: {e}")
    except Exception as e:
        print(f"[WARNING] Error di ANFIS, fallback ke FIS: {e}")

    # Fallback ke Fuzzy Inference System
    fuzzy_result = fuzzy_predict(input_data)
    return fuzzy_result["label"], fuzzy_result["score"], fuzzy_result["class_scores"], "Fuzzy Inference System (Fallback)"
