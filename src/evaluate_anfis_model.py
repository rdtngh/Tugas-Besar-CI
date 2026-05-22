"""
Evaluasi model Neuro-Fuzzy / ANFIS terhadap dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from joblib import load
from src.preprocessing import load_and_preprocess_data
from src.anfis_model import SimplifiedANFIS

MODEL_PATH = Path("models/anfis_model.pkl")
SCALER_PATH = Path("models/anfis_scaler.pkl")


def evaluate_anfis():
    print("=" * 70)
    print("EVALUASI NEURO-FUZZY / ANFIS")
    print("=" * 70)

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        print("[ERROR] Model ANFIS atau scaler belum tersedia.")
        print("Jalankan 'python -m src.train_anfis_model' terlebih dahulu.")
        return

    print("[1/4] Memuat model dan scaler...")
    model = SimplifiedANFIS.load(MODEL_PATH)
    scaler = load(SCALER_PATH)

    print("[2/4] Memuat dataset dan melakukan split...")
    _, X_test, _, y_test, _ = load_and_preprocess_data()
    y_true = y_test.values

    print("[3/4] Melakukan prediksi pada test set...")
    X_scaled = scaler.transform(X_test)
    _, probs = model.predict(X_scaled)
    y_pred = [int(np.argmax(p)) for p in probs]

    print("[4/4] Menghitung metrik evaluasi...")
    target_names = ['Rendah', 'Sedang', 'Tinggi']

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=[0, 1, 2], average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, labels=[0, 1, 2], average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=[0, 1, 2], average='weighted', zero_division=0)
    class_report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=target_names, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\n" + "=" * 70)
    print("HASIL EVALUASI ANFIS")
    print("=" * 70)
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)
    print(class_report)
    print("-" * 70)
    conf_df = pd.DataFrame(
        conf_matrix,
        index=['Actual: Rendah', 'Actual: Sedang', 'Actual: Tinggi'],
        columns=['Pred: Rendah', 'Pred: Sedang', 'Pred: Tinggi'],
    )
    print(conf_df)
    print("\n" + "=" * 70)
    print("Evaluasi selesai.")
    print("Model ini adalah ANFIS sederhana untuk tugas akademik.")


if __name__ == '__main__':
    import numpy as np
    evaluate_anfis()
