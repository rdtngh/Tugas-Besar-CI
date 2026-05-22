import streamlit as st
import pandas as pd
from src.predict import predict_stress
from src.recommendations import get_recommendation

# ============================================================================
# Configuration & Styling
# ============================================================================

st.set_page_config(
    page_title="Sistem Prediksi Tingkat Stres Mahasiswa",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk tampilan premium modern (Tanpa Emoticon & Kata-kata Asli)
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global font overwrite */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #0b0f19 !important;
        color: #f8fafc !important;
    }

    /* Card styling - Clean flat design with thin borders */
    .card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    .card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f3f4f6;
        margin-top: 0;
        margin-bottom: 16px;
    }

    /* Flat professional button design */
    .stButton > button {
        background: #4f46e5 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 12px 24px !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        transition: background-color 0.2s ease !important;
        width: 100%;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #4338ca !important;
        border: none !important;
    }

    /* Slider styling adjustments */
    .stSlider [role="slider"] {
        background: #4f46e5;
    }
    
    /* Minimalist metric card overrides */
    div[data-testid="stMetric"] {
        background: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 6px !important;
        padding: 14px !important;
    }

    /* Custom classes for badge text */
    .badge {
        background: #1f2937;
        color: #9ca3af;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid #374151;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================================
# Sidebar Configuration
# ============================================================================

with st.sidebar:
    st.markdown("### Informasi Proyek")
    st.divider()
    
    st.markdown("""
    **Nama Proyek:**
    Sistem Prediksi Tingkat Stres Mahasiswa
    
    **Mata Kuliah:**
    Kecerdasan Komputasional
    
    **Dataset:**
    StressLevelDataset.csv
    
    **Target Prediksi:**
    stress_level (Rendah/Sedang/Tinggi)
    """)
    
    st.divider()
    
    st.markdown("### Informasi Model")
    st.info("""
    **Status Model:**
    Fuzzy Inference System (FIS)
    
    **Metode:**
    Fuzzy Logic dengan Rule Base 14+ aturan
    
    **Pengembangan:**
    Siap dikembangkan ke Neuro-Fuzzy/ANFIS
    
    **Fitur Input:**
    - Tingkat Kecemasan
    - Kualitas Tidur
    - Beban Belajar
    - Performa Akademik
    - Dukungan Sosial
    - Kekhawatiran Karier
    """)

    st.divider()
    
    st.markdown("### Disclaimer")
    st.warning("""
    Sistem ini adalah **alat bantu prediksi awal** dan **BUKAN diagnosis medis**.
    Jika mengalami stress berkelanjutan, konsultasikan dengan profesional kesehatan mental.
    """)

# ============================================================================
# Header & Title
# ============================================================================

col_header = st.columns([1])
with col_header[0]:
    st.markdown("""
    <div style="text-align: left; padding: 20px 0 10px 0;">
        <h1 style="margin: 0; color: #ffffff; font-size: 2.25em; font-weight: 700; letter-spacing: -0.025em;">
            Sistem Prediksi Tingkat Stres Mahasiswa
        </h1>
        <p style="margin: 8px 0 16px 0; color: #9ca3af; font-size: 1.05em;">
            Prediksi awal tingkat stres berdasarkan faktor psikologis, akademik, sosial, dan gaya hidup
        </p>
        <div style="display: flex; gap: 8px; justify-content: flex-start;">
            <span class="badge">Neuro-Fuzzy</span>
            <span class="badge">ANFIS</span>
            <span class="badge">Student Stress</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================================================
# Main Layout: Two Columns
# ============================================================================

col_input, col_result = st.columns([1.2, 1.3], gap="large")

# ============================================================================
# LEFT COLUMN: INPUT FORM
# ============================================================================

with col_input:
    st.markdown("""
    <div class="card">
        <div class="card-title" style="margin: 0; color: #f1f5f9; font-size: 1.25rem;">Masukkan Data Mahasiswa</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Anxiety Level
    st.markdown("**1. Tingkat Kecemasan**")
    anxiety_level = st.slider(
        "Tingkat Kecemasan",
        min_value=0,
        max_value=20,
        value=10,
        step=1,
        label_visibility="collapsed",
        help="0 = Sangat tenang | 20 = Sangat cemas"
    )
    st.caption(f"Nilai: {anxiety_level}/20")
    
    st.divider()
    
    # Sleep Quality
    st.markdown("**2. Kualitas Tidur**")
    sleep_quality = st.slider(
        "Kualitas Tidur",
        min_value=0,
        max_value=5,
        value=3,
        step=1,
        label_visibility="collapsed",
        help="0 = Sangat buruk | 5 = Sangat baik"
    )
    st.caption(f"Nilai: {sleep_quality}/5")
    
    st.divider()
    
    # Study Load
    st.markdown("**3. Beban Belajar**")
    study_load = st.slider(
        "Beban Belajar",
        min_value=0,
        max_value=5,
        value=3,
        step=1,
        label_visibility="collapsed",
        help="0 = Sangat ringan | 5 = Sangat berat"
    )
    st.caption(f"Nilai: {study_load}/5")
    
    st.divider()
    
    # Academic Performance
    st.markdown("**4. Performa Akademik**")
    academic_performance = st.slider(
        "Performa Akademik",
        min_value=0,
        max_value=5,
        value=3,
        step=1,
        label_visibility="collapsed",
        help="0 = Sangat buruk | 5 = Sangat baik"
    )
    st.caption(f"Nilai: {academic_performance}/5")
    
    st.divider()
    
    # Social Support
    st.markdown("**5. Dukungan Sosial**")
    social_support = st.slider(
        "Dukungan Sosial",
        min_value=0,
        max_value=5,
        value=3,
        step=1,
        label_visibility="collapsed",
        help="0 = Sangat minim | 5 = Sangat banyak"
    )
    st.caption(f"Nilai: {social_support}/5")
    
    st.divider()
    
    # Future Career Concerns
    st.markdown("**6. Kekhawatiran Karier**")
    future_career_concerns = st.slider(
        "Kekhawatiran Karier",
        min_value=0,
        max_value=5,
        value=3,
        step=1,
        label_visibility="collapsed",
        help="0 = Tidak khawatir | 5 = Sangat khawatir"
    )
    st.caption(f"Nilai: {future_career_concerns}/5")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction Button
    predict_button = st.button("Prediksi Sekarang", use_container_width=True)

# ============================================================================
# RIGHT COLUMN: RESULTS
# ============================================================================

with col_result:
    if predict_button:
        # Prepare input data
        input_data = {
            "anxiety_level": anxiety_level,
            "sleep_quality": sleep_quality,
            "study_load": study_load,
            "academic_performance": academic_performance,
            "social_support": social_support,
            "future_career_concerns": future_career_concerns,
        }
        
        # Get prediction
        label, score, class_scores = predict_stress(input_data)
        recommendation = get_recommendation(label)
        
        # ====================================================================
        # RESULT DISPLAY
        # ====================================================================
        
        # Determine styling based on prediction
        if label == "Rendah":
            color = "#10b981"
            icon_text = "Status: Baik"
            container_func = st.success
        elif label == "Sedang":
            color = "#f59e0b"
            icon_text = "Status: Perhatian"
            container_func = st.warning
        else:  # Tinggi
            color = "#ef4444"
            icon_text = "Status: Tinggi"
            container_func = st.error
        
        # Main result card
        st.markdown(f"""
        <div class="card" style="border-top: 4px solid {color};">
            <div class="card-title" style="color: {color}; text-align: center; margin-bottom: 0; font-size: 1.25rem;">
                Hasil Prediksi
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Large prediction display
        col_pred1, col_pred2 = st.columns([1, 1])
        with col_pred1:
            container_func(f"### {label}")
        with col_pred2:
            st.metric("Skor Prediksi", f"{score:.0%}" if score else "N/A")
        
        st.divider()
        
        # Class Scores Visualization
        st.markdown("**Skor Prediksi per Kelas (Fuzzy Output):**")
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Buat bar chart untuk class scores
        fig, ax = plt.subplots(figsize=(8, 3.5))
        classes = list(class_scores.keys())
        scores = list(class_scores.values())
        colors = ['#10b981', '#f59e0b', '#ef4444']
        
        bars = ax.barh(classes, scores, color=colors, alpha=0.85, edgecolor='#374151', linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Firing Strength', color='#9ca3af', fontsize=9)
        ax.set_facecolor('#111827')
        fig.patch.set_facecolor('#0b0f19')
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#1f2937')
        ax.spines['bottom'].set_color('#1f2937')
        ax.tick_params(colors='#9ca3af', labelsize=9)
        
        # Add value labels on bars
        for i, (bar, score_val) in enumerate(zip(bars, scores)):
            ax.text(score_val + 0.02, i, f'{score_val:.3f}', 
                    va='center', color='#ffffff', fontsize=9, fontweight='500')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.divider()
        
        # Progress bar visualization
        st.markdown("**Tingkat Keyakinan Prediksi (Maximum Membership):**")
        if score:
            st.progress(score, text=f"{score:.1%}")
        
        st.divider()
        
        # ====================================================================
        # RECOMMENDATION CARD
        # ====================================================================
        
        st.markdown(f"""
        <div class="card" style="border-left: 4px solid {color}; background: #111827;">
            <div class="card-title" style="color: {color}; font-size: 1.15rem; margin-bottom: 10px;">Rekomendasi</div>
            <p style="color: #cbd5e1; line-height: 1.6; margin: 0; font-size: 0.95rem;">
                {recommendation}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # ====================================================================
        # INPUT DETAIL TABLE
        # ====================================================================
        
        st.markdown("### Detail Input Anda")
        
        detail_df = pd.DataFrame({
            "Faktor": [
                "Tingkat Kecemasan",
                "Kualitas Tidur",
                "Beban Belajar",
                "Performa Akademik",
                "Dukungan Sosial",
                "Kekhawatiran Karier"
            ],
            "Nilai": [
                f"{anxiety_level}/20",
                f"{sleep_quality}/5",
                f"{study_load}/5",
                f"{academic_performance}/5",
                f"{social_support}/5",
                f"{future_career_concerns}/5"
            ]
        })
        
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    else:
        # Placeholder when no prediction yet
        st.markdown("""
        <div class="card" style="text-align: center; padding: 50px 20px; border-style: dashed; border-color: #374151;">
            <div style="color: #cbd5e1; font-size: 1.15rem; font-weight: 500; margin-bottom: 8px;">
                Hasil Prediksi
            </div>
            <p style="color: #6b7280; margin: 0; font-size: 0.95rem;">
                Isi data di sebelah kiri dan klik "Prediksi Sekarang" untuk melihat hasil prediksi tingkat stres Anda.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# BOTTOM SECTION: How It Works
# ============================================================================

st.divider()

st.markdown("""
<div style="text-align: center; margin: 30px 0 10px 0;">
    <h2 style="color: #ffffff; font-size: 1.5em; font-weight: 600;">Bagaimana Sistem Bekerja?</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5, gap="small")

steps = [
    ("Input", "Data dari 6 fitur mahasiswa"),
    ("Fuzzifikasi", "Konversi ke membership fuzzy"),
    ("Rule Base", "Evaluasi 14+ aturan IF-THEN"),
    ("Inferensi", "Hitung firing strength"),
    ("Defuzzifikasi", "Output: Rendah/Sedang/Tinggi"),
]

for idx, (col, (title, desc)) in enumerate(zip([col1, col2, col3, col4, col5], steps)):
    with col:
        st.markdown(f"""
        <div class="card" style="text-align: center; min-height: 140px; display: flex; flex-direction: column; justify-content: center; padding: 15px;">
            <div style="font-size: 1.05rem; font-weight: 600; color: #4f46e5; margin-bottom: 6px;">{title}</div>
            <p style="font-size: 0.85rem; color: #9ca3af; margin: 0; line-height: 1.4;">
                {desc}
            </p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div style="text-align: center; color: #4b5563; font-size: 0.85rem; padding: 10px 0;">
    <p style="margin: 4px 0;">
        Tugas Besar Kecerdasan Komputasional | Sistem Prediksi Tingkat Stres Mahasiswa
    </p>
    <p style="margin: 4px 0; color: #374151;">
        Sistem ini hanya alat bantu prediksi awal dan BUKAN diagnosis medis profesional.
    </p>
</div>
""", unsafe_allow_html=True)