# Sistem Prediksi Tingkat Stres Mahasiswa Menggunakan Neuro-Fuzzy

Sistem ini adalah aplikasi interaktif berbasis web untuk memprediksi tingkat stres mahasiswa secara awal. Proyek dibuat sebagai tugas besar mata kuliah Kecerdasan Komputasional menggunakan pendekatan machine learning dengan kerangka kerja yang dapat dikembangkan ke Neuro-Fuzzy / ANFIS.

## Tujuan Sistem
- Menyediakan alat bantu prediksi awal tingkat stres mahasiswa.
- Menampilkan rekomendasi sederhana berdasarkan hasil prediksi.
- Menjaga antarmuka yang mudah digunakan dan informatif.

## Dataset yang Digunakan
- `data/StressLevelDataset.csv`
- Target prediksi: `stress_level`
- Fitur input utama:
  - `anxiety_level`
  - `sleep_quality`
  - `study_load`
  - `academic_performance`
  - `social_support`
  - `future_career_concerns`

## Output Sistem
- Prediksi tingkat stres:
  - `Rendah`
  - `Sedang`
  - `Tinggi`
- Rekomendasi sederhana berdasarkan hasil prediksi.

## Teknologi yang Digunakan
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Struktur Folder
```
Tugas-Besar-CI/
+-- app.py
+-- README.md
+-- requirements.txt
+-- .gitignore
+-- data/
�   +-- StressLevelDataset.csv
+-- models/
�   +-- stress_model.pkl
�   +-- scaler.pkl
+-- notebooks/
�   +-- exploratory_analysis.ipynb
+-- src/
�   +-- __init__.py
�   +-- preprocessing.py
�   +-- train_model.py
�   +-- predict.py
�   +-- recommendations.py
+-- docs/
    +-- screenshots/
```

## Cara Menjalankan Project
1. Instal dependensi:
```bash
pip install -r requirements.txt
```
2. Jalankan aplikasi Streamlit:
```bash
streamlit run app.py
```
atau:
```bash
python -m streamlit run app.py
```

## Cara Training Model
1. Pastikan dataset `data/StressLevelDataset.csv` tersedia.
2. Jalankan script training sebagai modul Python:
```bash
python -m src.train_model
```
3. Model akan disimpan di folder `models/` sebagai `stress_model.pkl` dan `scaler.pkl`.

## Catatan
- Saat ini model yang digunakan adalah baseline `RandomForestClassifier` untuk memastikan aplikasi dapat berjalan.
- Model ini dapat dikembangkan di masa depan menjadi metode Neuro-Fuzzy / ANFIS.

## Cara Kerja Sistem Singkat
1. Pengguna memasukkan nilai fitur melalui slider di aplikasi.
2. Sistem memproses input dan memuat model jika tersedia.
3. Sistem menghasilkan prediksi tingkat stres dan rekomendasi.
4. Jika model belum tersedia, aplikasi menggunakan prediksi fallback berbasis skor sederhana.

## Anggota Kelompok
- Najlatika (123140078)
- Bening Apni Prameswari (123140089)
- Raditya Alrasyid Nugroho (123140125)
- Raisya Syifa Saleh (123140169)
- Muhamad Arif Ardani (123140186)
