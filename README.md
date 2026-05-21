# Sistem Prediksi Tingkat Stres Mahasiswa Menggunakan Metode Neuro-Fuzzy

Sistem ini adalah aplikasi interaktif berbasis web untuk memprediksi tingkat stres mahasiswa secara awal. Proyek dibuat sebagai tugas besar mata kuliah Kecerdasan Komputasional menggunakan pendekatan Fuzzy Inference System (FIS) sebagai fondasi pengembangan Neuro-Fuzzy/ANFIS.

## Tujuan Sistem
- Menyediakan alat bantu prediksi awal tingkat stres mahasiswa berbasis logika fuzzy.
- Menampilkan rekomendasi berdasarkan hasil prediksi.
- Menyediakan interpretabilitas tinggi melalui rule base yang jelas.
- Membentuk dasar untuk pengembangan sistem Neuro-Fuzzy/ANFIS yang adaptif.

## Dataset yang Digunakan
- **File:** `data/StressLevelDataset.csv`
- **Sumber:** Kaggle Public Dataset
- **Target Prediksi:** `stress_level` (0=Rendah, 1=Sedang, 2=Tinggi)
- **Fitur Input Utama:**
  - `anxiety_level` (0-20): Tingkat kecemasan
  - `sleep_quality` (0-5): Kualitas tidur
  - `study_load` (0-5): Beban belajar
  - `academic_performance` (0-5): Performa akademik
  - `social_support` (0-5): Dukungan sosial
  - `future_career_concerns` (0-5): Kekhawatiran karier

## Output Sistem
- **Prediksi Tingkat Stres:**
  - `Rendah` (hijau) - Stres terkontrol
  - `Sedang` (kuning) - Memerlukan perhatian
  - `Tinggi` (merah) - Memerlukan intervensi
- **Skor Prediksi:** Confidence score (0-1) dari firing strength maksimum
- **Class Scores:** Firing strength untuk setiap kelas fuzzy
- **Rekomendasi:** Saran personal berdasarkan hasil prediksi
- **Visualisasi:** Bar chart class scores dan progress bar

## Teknologi yang Digunakan
- **Python 3.10+**
- **Streamlit:** UI/UX interaktif
- **Pandas & NumPy:** Data processing
- **Scikit-learn:** Preprocessing dan evaluasi
- **Matplotlib:** Visualisasi
- **Fuzzy Logic:** Membership functions dan rule base

## Metode Neuro-Fuzzy

### Fuzzy Inference System (FIS)
Sistem ini mengimplementasikan Fuzzy Inference System (Mamdani) dengan 5 tahap:

#### 1. **Fuzzifikasi (Fuzzification)**
Mengubah input numerik menjadi derajat keanggotaan (membership degree) terhadap himpunan fuzzy.

**Membership Functions:**
- Triangular: µ(x) berbentuk segitiga dengan puncak di tengah
- Trapezoidal: µ(x) berbentuk trapesium dengan plateau di tengah

**Himpunan Fuzzy yang Didefinisikan:**
```
anxiety_level       : rendah | sedang | tinggi
sleep_quality       : buruk | sedang | baik
study_load          : ringan | sedang | berat
academic_performance: rendah | sedang | tinggi
social_support      : rendah | sedang | tinggi
future_career_concerns: rendah | sedang | tinggi
```

#### 2. **Rule Base (14+ Aturan IF-THEN)**
Aturan dievaluasi dengan operator AND (minimum):

**Aturan Stress Level TINGGI (6 rules):**
- IF anxiety_level = tinggi AND sleep_quality = buruk THEN stress_level = tinggi
- IF anxiety_level = tinggi AND study_load = berat THEN stress_level = tinggi
- IF anxiety_level = tinggi AND future_career_concerns = tinggi THEN stress_level = tinggi
- IF study_load = berat AND academic_performance = rendah THEN stress_level = tinggi
- IF social_support = rendah AND anxiety_level = tinggi THEN stress_level = tinggi
- IF sleep_quality = buruk AND future_career_concerns = tinggi THEN stress_level = tinggi

**Aturan Stress Level SEDANG (4 rules):**
- IF anxiety_level = sedang AND study_load = sedang THEN stress_level = sedang
- IF sleep_quality = sedang AND academic_performance = sedang THEN stress_level = sedang
- IF future_career_concerns = sedang AND social_support = sedang THEN stress_level = sedang
- IF study_load = sedang AND sleep_quality = sedang THEN stress_level = sedang

**Aturan Stress Level RENDAH (4 rules):**
- IF anxiety_level = rendah AND sleep_quality = baik THEN stress_level = rendah
- IF social_support = tinggi AND academic_performance = tinggi THEN stress_level = rendah
- IF study_load = ringan AND future_career_concerns = rendah THEN stress_level = rendah
- IF anxiety_level = rendah AND social_support = tinggi THEN stress_level = rendah

#### 3. **Inference Engine**
Menghitung firing strength untuk setiap aturan menggunakan operator AND (minimum) untuk antecedent.

#### 4. **Aggregation**
Mengagregasi semua aturan untuk mendapatkan skor kelas (maximum dari setiap kelas):
```
score_rendah = max(firing_strength dari semua aturan output "rendah")
score_sedang = max(firing_strength dari semua aturan output "sedang")
score_tinggi = max(firing_strength dari semua aturan output "tinggi")
```

#### 5. **Defuzzifikasi (Defuzzification)**
Mengambil kelas dengan skor tertinggi sebagai output final (maximum membership).

### Pengembangan Neuro-Fuzzy/ANFIS
Sistem ini dirancang untuk mudah dikembangkan ke Neuro-Fuzzy/ANFIS dengan menambahkan:

1. **Adaptive Membership Functions:** Parameter membership function dapat dipelajari dari data
2. **Learning Mechanism:** Gradient descent atau least squares estimation
3. **Rule Optimization:** Jumlah dan bentuk aturan dapat disesuaikan
4. **Hybrid Learning:** Forward pass (LSE untuk consequent) + Backward pass (gradient descent untuk premise)

## Struktur Folder
```
Tugas-Besar-CI/
+-- app.py                          # Aplikasi Streamlit utama
+-- README.md                       # Dokumentasi project
+-- requirements.txt                # Dependencies Python
+-- .gitignore                      # Git ignore rules
+-- data/
¦   +-- StressLevelDataset.csv     # Dataset Kaggle
+-- models/
¦   +-- fuzzy_config.json          # Konfigurasi FIS
¦   +-- stress_model.pkl           # Model RandomForest (opsional)
¦   +-- scaler.pkl                 # Data scaler (opsional)
+-- notebooks/
¦   +-- exploratory_analysis.ipynb # Exploratory data analysis
+-- src/
¦   +-- __init__.py
¦   +-- preprocessing.py            # Load & preprocess dataset
¦   +-- fuzzy_membership.py         # Membership functions (triangular, trapezoidal)
¦   +-- fuzzy_rules.py              # Rule base 14+ aturan
¦   +-- fuzzy_model.py              # FIS engine (fuzzify -> infer -> defuzzify)
¦   +-- train_fuzzy_model.py        # Setup & dokumentasi FIS
¦   +-- evaluate_model.py           # Evaluasi model terhadap dataset
¦   +-- predict.py                  # Fungsi prediksi untuk app
¦   +-- train_model.py              # Training model ML (RandomForest)
¦   +-- recommendations.py          # Rekomendasi berdasarkan hasil prediksi
+-- docs/
    +-- screenshots/                # Tangkapan layar aplikasi
```

## Cara Menjalankan Project

### 1. Setup Awal
```bash
# Clone atau extract project
cd Tugas-Besar-CI

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Menjalankan Aplikasi Streamlit
```bash
# Opsi 1 (recommended)
streamlit run app.py

# Opsi 2
python -m streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`

### 3. Evaluasi Model Fuzzy
```bash
python -m src.evaluate_model
```

Output akan menampilkan:
- Accuracy, Precision, Recall, F1-Score
- Classification Report
- Confusion Matrix
- Rekomendasi pengembangan

### 4. Setup/Training Fuzzy System
```bash
python -m src.train_fuzzy_model
```

Output akan menyimpan konfigurasi FIS ke `models/fuzzy_config.json`

### 5. Training Model ML Baseline (RandomForest)
```bash
python -m src.train_model
```

Opsional untuk evaluasi tambahan dengan model machine learning.

## Cara Kerja Sistem Singkat

```
INPUT DATA (6 fitur)
       ?
FUZZIFIKASI (Membership Functions)
       ?
EVALUASI RULE BASE (14+ aturan IF-THEN)
       ?
INFERENCE (Min-Max)
       ?
AGGREGATION (Maksimum per kelas)
       ?
DEFUZZIFIKASI (Pilih kelas tertinggi)
       ?
OUTPUT: Label + Score + Rekomendasi
```

## Evaluasi Model

Model telah dievaluasi menggunakan metrik performa:
- **Accuracy:** Persentase prediksi yang benar dari total sampel
- **Precision:** Akurasi pada kelas yang diprediksi (TP / (TP + FP))
- **Recall:** Kemampuan menemukan semua instans kelas sebenarnya (TP / (TP + FN))
- **F1-Score:** Rata-rata harmonis precision dan recall
- **Confusion Matrix:** Breakdown detail prediksi per kelas

Jalankan evaluasi: `python -m src.evaluate_model`

## File-File Penting

| File | Fungsi |
|------|--------|
| `src/fuzzy_membership.py` | Triangular & trapezoidal membership functions |
| `src/fuzzy_rules.py` | 14+ rule base dengan operator AND (min) |
| `src/fuzzy_model.py` | Implementasi FIS engine (fuzzify ? infer ? defuzzify) |
| `src/evaluate_model.py` | Evaluasi performa model terhadap dataset |
| `src/predict.py` | Fungsi prediksi yang digunakan app.py |
| `app.py` | Aplikasi Streamlit dengan UI modern |

## Catatan Penting

- **Bukan Diagnosis Medis:** Sistem ini hanya alat bantu prediksi awal, BUKAN diagnosis profesional.
- **Fuzzy Logic Foundation:** Model saat ini menggunakan Fuzzy Inference System berbasis aturan.
- **Pengembangan ANFIS:** Sistem dirancang untuk mudah dikembangkan ke ANFIS dengan learning mechanism.
- **Data Kaggle:** Dataset publik dari Kaggle untuk penelitian akademis.
- **Interpretabilitas:** Rule base fuzzy mudah dipahami dan dijelaskan (white-box model).

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
