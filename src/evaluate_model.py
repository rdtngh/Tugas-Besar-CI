"""
Legacy baseline evaluation untuk Fuzzy Inference System terhadap dataset.

Script ini mengevaluasi model FIS lama sebagai pembanding untuk model Neuro-Fuzzy / ANFIS.
File ini tidak dipanggil langsung oleh app.py, melainkan hanya untuk analisis dan perbandingan.
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.preprocessing import load_and_preprocess_data
from src.fuzzy_model import fuzzy_predict_batch


def load_raw_dataset():
    """Load dataset tanpa normalisasi untuk prediksi."""
    csv_path = "data/StressLevelDataset.csv"
    df = pd.read_csv(csv_path)
    return df


def map_target_to_label(y):
    """
    Mapping target stress_level ke label.
    
    Jika input angka (0, 1, 2): mapping ke Rendah, Sedang, Tinggi
    Jika input string: normalisasi ke format standar
    """
    label_map = {
        0: 'Rendah',
        1: 'Sedang',
        2: 'Tinggi',
        'rendah': 'Rendah',
        'sedang': 'Sedang',
        'tinggi': 'Tinggi',
        'Rendah': 'Rendah',
        'Sedang': 'Sedang',
        'Tinggi': 'Tinggi',
    }
    
    if isinstance(y, (list, np.ndarray, pd.Series)):
        return [label_map.get(val, 'Sedang') for val in y]
    return label_map.get(y, 'Sedang')


def evaluate_fuzzy_model():
    """
    Evaluasi Fuzzy Inference System terhadap seluruh dataset.
    """
    print("=" * 70)
    print("EVALUASI LEGACY FIS BASELINE - Prediksi Tingkat Stres Mahasiswa")
    print("=" * 70)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    try:
        df = load_raw_dataset()
        
        # Validasi fitur dan target
        required_features = [
            'anxiety_level', 'sleep_quality', 'study_load',
            'academic_performance', 'social_support', 'future_career_concerns'
        ]
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f"  [ERROR] Fitur tidak ditemukan: {missing_features}")
            return
        
        if 'stress_level' not in df.columns:
            print("  [ERROR] Kolom target 'stress_level' tidak ditemukan")
            return
        
        X = df[required_features].copy()
        y_true = df['stress_level'].copy()
        
        print(f"  [OK] Dataset loaded: {len(df)} sampel")
        print(f"  [OK] Fitur: {required_features}")
        print(f"  [OK] Target: stress_level")
    
    except Exception as e:
        print(f"  [ERROR] Error loading dataset: {e}")
        return
    
    # Handle missing values
    print("\n[2/4] Preprocessing data...")
    try:
        # PERBAIKAN: Menggunakan median statis atau fallback yang aman dari rule base awal
        X = X.fillna(X.median())
        y_true = map_target_to_label(y_true)
        print(f"  [OK] Missing values handled")
    except Exception as e:
        print(f"  [ERROR] Error preprocessing: {e}")
        return
    
    # Prediksi menggunakan Fuzzy Model
    print("\n[3/4] Melakukan prediksi dengan Fuzzy Inference System...")
    try:
        predictions = fuzzy_predict_batch(X)
        y_pred = [p['label'] for p in predictions]
        print(f"  [OK] Prediksi selesai untuk {len(y_pred)} sampel")
    except Exception as e:
        print(f"  [ERROR] Error dalam prediksi: {e}")
        return
    
    # Hitung metrik evaluasi
    print("\n[4/4] Menghitung metrik evaluasi...")
    try:
        # Target urutan kelas yang konsisten secara skalar ekonomi fuzzy
        target_labels = ['Rendah', 'Sedang', 'Tinggi']
        
        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1
        # PERBAIKAN: Menambahkan parameter labels eksplisit agar kalkulasi pembobotan rata-rata linear tidak bergeser
        precision = precision_score(y_true, y_pred, labels=target_labels, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, labels=target_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=target_labels, average='weighted', zero_division=0)
        
        # Classification Report
        # PERBAIKAN: Memasang pasangan parameter labels dan target_names secara eksplisit bersesuaian
        class_report = classification_report(
            y_true, y_pred, 
            labels=target_labels,
            target_names=target_labels,
            zero_division=0
        )
        
        # Confusion Matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=target_labels)
        
        print("  [OK] Metrik evaluasi dihitung")
    
    except Exception as e:
        print(f"  [ERROR] Error menghitung metrik: {e}")
        return
    
    # Display hasil evaluasi
    print("\n" + "=" * 70)
    print("HASIL EVALUASI")
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
    print("CONFUSION MATRIX")
    print("-" * 70)
    conf_df = pd.DataFrame(
        conf_matrix,
        index=['Actual: Rendah', 'Actual: Sedang', 'Actual: Tinggi'],
        columns=['Pred: Rendah', 'Pred: Sedang', 'Pred: Tinggi']
    )
    print(conf_df)
    
    print("\n" + "=" * 70)
    print("KESIMPULAN")
    print("=" * 70)
    print("""
    Legacy FIS baseline telah dievaluasi terhadap dataset.
    
    Interpretasi Hasil:
    - Accuracy: Persentase prediksi yang benar
    - Precision: Akurasi pada kelas yang diprediksi
    - Recall: Kemampuan menemukan semua instans kelas sebenarnya
    - F1-Score: Rata-rata harmonis precision dan recall
    
    Catatan penting:
    - Hasil ini adalah baseline FIS, bukan model utama.
    - Fokus utama proyek adalah Neuro-Fuzzy / ANFIS.
    - Jalankan 'python -m src.evaluate_anfis_model' untuk evaluasi ANFIS.
    - FIS hanya digunakan jika ANFIS belum tersedia atau sebagai perbandingan.
    """)
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_fuzzy_model()