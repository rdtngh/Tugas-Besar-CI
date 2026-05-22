import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.preprocessing import load_and_preprocess_data

MODEL_PATH = Path("models/stress_model.pkl")
SCALER_PATH = Path("models/scaler.pkl")


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    # Model baseline legacy menggunakan RandomForestClassifier.
    # Proyek utama berfokus pada Neuro-Fuzzy / ANFIS, bukan pada model ini.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Evaluasi Model Baseline")
    print("------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model tersimpan di: {MODEL_PATH}")
    print(f"Scaler tersimpan di: {SCALER_PATH}")


if __name__ == "__main__":
    main()
