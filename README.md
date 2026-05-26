# Sistem Prediksi Tingkat Stres Mahasiswa dengan Neuro-Fuzzy / ANFIS

Sistem ini adalah aplikasi interaktif berbasis web untuk memprediksi tingkat stres mahasiswa. Proyek dikembangkan sebagai tugas besar mata kuliah Kecerdasan Komputasional dengan fokus utama pada pendekatan Neuro-Fuzzy / ANFIS, sementara FIS tetap disimpan sebagai baseline dan fallback.

## Tujuan Sistem
- Memprediksi tingkat stres mahasiswa dalam 3 kelas: `Rendah`, `Sedang`, `Tinggi`.
- Menggunakan membership Gaussian adaptif yang bisa dipelajari dari data.
- Menyediakan confidence score dan class score untuk hasil prediksi.
- Menjaga FIS lama sebagai pembanding dan fallback ketika ANFIS belum tersedia.
- ANFIS adalah pendekatan hibrida ANN + Fuzzy sederhana untuk kebutuhan akademik.
- Streamlit digunakan sebagai antarmuka interaktif bagi pengguna.

## Dataset
- **File:** `data/StressLevelDataset.csv`
- **Target:** `stress_level`
- **Target Output:**
  - `0` = Rendah
  - `1` = Sedang
  - `2` = Tinggi

## Fitur Input Utama
- `anxiety_level` (0-20)
- `sleep_quality` (0-5)
- `study_load` (0-5)
- `academic_performance` (0-5)
- `social_support` (0-5)
- `future_career_concerns` (0-5)

## Alur Metode Simplified Neuro-Fuzzy / ANFIS
1. Input data mahasiswa dimasukkan ke sistem.
2. Data dinormalisasi menggunakan `MinMaxScaler`.
3. Setiap fitur difuzzifikasi dengan membership Gaussian adaptif.
4. Rule layer menghitung firing strength untuk setiap term fuzzy.
5. Firing strength dinormalisasi untuk kontribusi setiap term.
6. Consequent layer linear menghasilkan skor kelas numerik.
7. Softmax menghasilkan probabilitas kelas akhir.
8. Output akhir adalah label `Rendah`, `Sedang`, atau `Tinggi`, beserta confidence score.

Sistem menggunakan pendekatan simplified Neuro-Fuzzy/ANFIS sebagai model utama dan Fuzzy Inference System sebagai fallback.

## Perbedaan FIS vs Neuro-Fuzzy
- **FIS biasa:**
  - Membership function statis (triangular/trapezoidal).
  - Rule base manual dengan inferensi Mamdani.
  - Output ditentukan dari skor fuzzy maksimum.
- **Neuro-Fuzzy / ANFIS:**
  - Implementasi simplified ANFIS untuk kebutuhan akademik.
  - Membership Gaussian adaptif yang dilatih dari data.
  - Learning dengan gradient descent sederhana.
  - Consequent layer linear dan softmax untuk probabilitas.
  - FIS lama tetap menjadi baseline dan fallback.

## Struktur Folder
```
Tugas-Besar-CI/
├─ app.py
├─ README.md
├─ requirements.txt
├─ data/
│  └─ StressLevelDataset.csv
├─ models/
│  ├─ fuzzy_config.json
│  ├─ stress_model.pkl   # Legacy baseline RandomForest
│  ├─ scaler.pkl         # Legacy baseline scaler
│  ├─ anfis_model.pkl
│  └─ anfis_scaler.pkl
├─ notebooks/
│  └─ exploratory_analysis.ipynb
└─ src/
   ├─ __init__.py
   ├─ preprocessing.py
   ├─ fuzzy_membership.py
   ├─ fuzzy_rules.py
   ├─ fuzzy_model.py
   ├─ train_fuzzy_model.py
   ├─ evaluate_model.py
   ├─ predict.py
   ├─ train_model.py
   ├─ recommendations.py
   ├─ anfis_model.py
   ├─ train_anfis_model.py
   └─ evaluate_anfis_model.py
```

## Cara Menjalankan Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Latih Model Neuro-Fuzzy / ANFIS
```bash
python -m src.train_anfis_model
```

### 3. Evaluasi Model ANFIS
```bash
python -m src.evaluate_anfis_model
```

### 4. Menjalankan Aplikasi Streamlit
```bash
streamlit run app.py
```

### 5. Evaluasi FIS Baseline (opsional)
```bash
python -m src.evaluate_model
```

## Evaluasi Model
Evaluasi utama proyek ini adalah terhadap model Neuro-Fuzzy / ANFIS.
Setelah training ANFIS, skrip evaluasi menampilkan:
- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

FIS lama hanya berfungsi sebagai baseline perbandingan dan fallback ketika model ANFIS belum tersedia.

## Penjelasan Singkat Arsitektur
- `src/anfis_model.py`: simplified Neuro-Fuzzy model dengan membership Gaussian adaptif.
- `src/train_anfis_model.py`: pelatihan ANFIS dari dataset.
- `src/evaluate_anfis_model.py`: evaluasi performa ANFIS.
- `src/predict.py`: prediksi ANFIS + fallback FIS untuk Streamlit.
- `src/fuzzy_model.py`: FIS lama sebagai baseline.
- `app.py`: aplikasi Streamlit utama.

## Catatan Penting
- **Bukan diagnosis medis:** Sistem ini hanya untuk prediksi awal.
- **Neuro-Fuzzy:** Model ini adalah simplified ANFIS untuk kebutuhan akademik.
- **FIS lama tetap ada:** Digunakan sebagai baseline dan fallback berbasis rule IF-THEN.
- **Dataset:** Digunakan untuk training dan evaluasi model.
- **Streamlit:** Menyediakan antarmuka interaktif untuk pengguna.
- **Output:** prediksi `Rendah`, `Sedang`, `Tinggi` + confidence score.

## Anggota Kelompok
- Najlatika (123140078)
- Bening Apni Prameswari (123140089)
- Raditya Alrasyid Nugroho (123140125)
- Raisya Syifa Saleh (123140169)
- Muhamad Arif Ardani (123140186)

## Referensi & Resource

- **Fuzzy Logic:** https://en.wikipedia.org/wiki/Fuzzy_logic
- **ANFIS:** https://en.wikipedia.org/wiki/Adaptive_neuro_fuzzy_inference_system
- **Streamlit Documentation:** https://docs.streamlit.io/
- **Scikit-learn:** https://scikit-learn.org/
- **Dataset:** Kaggle StressLevelDataset (https://www.kaggle.com/datasets/saurabhshur/student-stress-factors-dataset)

## Lisensi & Catatan

Proyek ini dibuat untuk keperluan akademis sebagai tugas besar mata kuliah **Kecerdasan Komputasional** semester 6. Dataset berasal dari Kaggle dan bersifat publik untuk penelitian.
