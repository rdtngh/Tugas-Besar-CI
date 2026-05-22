"""
Script training ANFIS sederhana untuk prediksi stres mahasiswa.
"""

from pathlib import Path
from joblib import dump
from src.preprocessing import load_and_preprocess_data
from src.anfis_model import SimplifiedANFIS

MODEL_PATH = Path("models/anfis_model.pkl")
SCALER_PATH = Path("models/anfis_scaler.pkl")


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[1/4] Memuat dan memproses dataset...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print(f"  - Data train: {len(X_train)} sampel")
    print(f"  - Data test: {len(X_test)} sampel")

    print("[2/4] Membangun model Neuro-Fuzzy/ANFIS sederhana...")
    anfis_model = SimplifiedANFIS(input_dim=X_train.shape[1], num_terms=3, num_classes=3)

    print("[3/4] Melatih model ANFIS...")
    anfis_model.train(
        X_train.values,
        y_train.values,
        epochs=250,
        lr=0.03,
        batch_size=32,
        verbose=True,
    )

    print("[4/4] Menyimpan model dan scaler...")
    anfis_model.save(MODEL_PATH)
    dump(scaler, SCALER_PATH)

    print("\nTraining selesai.")
    print(f"Model ANFIS tersimpan di: {MODEL_PATH}")
    print(f"Scaler tersimpan di: {SCALER_PATH}")
    print("Jalankan 'python -m src.evaluate_anfis_model' untuk mengevaluasi performa.")


if __name__ == '__main__':
    main()
