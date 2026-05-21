import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

FEATURES = [
    "anxiety_level",
    "sleep_quality",
    "study_load",
    "academic_performance",
    "social_support",
    "future_career_concerns",
]
TARGET = "stress_level"


def load_and_preprocess_data(csv_path: str = "data/StressLevelDataset.csv", test_size: float = 0.2, random_state: int = 42):
    """Membaca dataset, membersihkan missing value, menormalisasi fitur, dan membagi data menjadi train/test."""
    data_path = Path(csv_path)
    df = pd.read_csv(data_path)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    if X.isnull().any().any() or y.isnull().any():
        X = X.fillna(X.median())
        y = y.fillna(y.mode().iloc[0])

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, scaler
