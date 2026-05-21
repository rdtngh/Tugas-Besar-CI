import streamlit as st
import pandas as pd
from src.predict import predict_stress
from src.recommendations import get_recommendation

# ============================================================================
# Configuration & Styling
# ============================================================================

st.set_page_config(
    page_title="Sistem Prediksi Tingkat Stres Mahasiswa",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk tampilan modern
custom_css = """
<style>
    /* Global styling */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --border-color: #334155;
    }

    /* Background */
    body {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%);
        color: var(--text-primary);
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%);
    }

    /* Text styling */
    h1, h2, h3 {
        color: var(--text-primary) !important;
    }

    /* Card styling */
    .card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(99, 102, 241, 0.4);
    }

    /* Slider styling */
    .stSlider [role="slider"] {
        background: var(--primary-color);
    }

    /* Metric cards */
    .stMetric {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================================
# Sidebar Configuration
# ============================================================================

with st.sidebar:
    st.markdown("### 📋 Informasi Proyek")
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
    
    st.markdown("### 🔧 Informasi Model")
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
    
    st.markdown("### ⚠️ Disclaimer")
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
    <div style="text-align: center; padding: 30px 0;">
        <h1 style="margin: 0; color: #f1f5f9; font-size: 2.5em;">🧠 Sistem Prediksi Tingkat Stres Mahasiswa</h1>
        <p style="margin: 10px 0; color: #cbd5e1; font-size: 1.1em;">
            Prediksi awal tingkat stres berdasarkan faktor psikologis, akademik, sosial, dan gaya hidup
        </p>
        <div style="display: flex; gap: 10px; justify-content: center; margin-top: 15px;">
            <span style="background: #6366f1; color: white; padding: 6px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600;">Neuro-Fuzzy</span>
            <span style="background: #8b5cf6; color: white; padding: 6px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600;">ANFIS</span>
            <span style="background: #10b981; color: white; padding: 6px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600;">Student Stress</span>
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
        <h2 style="margin: 0 0 20px 0; color: #f1f5f9;">📝 Masukkan Data Mahasiswa</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Anxiety Level
    st.markdown("**1. Tingkat Kecemasan** 😰")
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
    st.markdown("**2. Kualitas Tidur** 😴")
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
    st.markdown("**3. Beban Belajar** 📚")
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
    st.markdown("**4. Performa Akademik** 🎓")
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
    st.markdown("**5. Dukungan Sosial** 👥")
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
    st.markdown("**6. Kekhawatiran Karier** 🚀")
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
    predict_button = st.button("🔮 Prediksi Sekarang", use_container_width=True)

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
            emoji = "✅"
            icon_text = "Status: Baik"
            container_func = st.success
        elif label == "Sedang":
            color = "#f59e0b"
            emoji = "⚠️"
            icon_text = "Status: Perhatian"
            container_func = st.warning
        else:  # Tinggi
            color = "#ef4444"
            emoji = "🚨"
            icon_text = "Status: Tinggi"
            container_func = st.error
        
        # Main result card
        st.markdown(f"""
        <div class="card" style="border-left: 5px solid {color};">
            <h2 style="text-align: center; color: {color}; margin: 0 0 10px 0;">
                {emoji} Hasil Prediksi
            </h2>
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
        fig, ax = plt.subplots(figsize=(8, 4))
        classes = list(class_scores.keys())
        scores = list(class_scores.values())
        colors = ['#10b981', '#f59e0b', '#ef4444']
        
        bars = ax.barh(classes, scores, color=colors, alpha=0.7, edgecolor='#334155', linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Firing Strength', color='#cbd5e1', fontsize=10)
        ax.set_facecolor('#1e293b')
        fig.patch.set_facecolor('#0f172a')
        
        # Styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        ax.spines['bottom'].set_color('#334155')
        ax.tick_params(colors='#cbd5e1')
        
        # Add value labels on bars
        for i, (bar, score_val) in enumerate(zip(bars, scores)):
            ax.text(score_val + 0.02, i, f'{score_val:.3f}', 
                   va='center', color='#cbd5e1', fontsize=10, fontweight='bold')
        
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
        <div class="card" style="border-left: 5px solid {color};">
            <h3 style="color: {color}; margin: 0 0 15px 0;">💡 Rekomendasi</h3>
            <p style="color: #cbd5e1; line-height: 1.6; margin: 0;">
                {recommendation}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # ====================================================================
        # INPUT DETAIL TABLE
        # ====================================================================
        
        st.markdown("### 📊 Detail Input Anda")
        
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
        <div class="card" style="text-align: center; padding: 40px 20px;">
            <h3 style="color: #cbd5e1; margin: 0 0 10px 0;">
                📋 Hasil Prediksi
            </h3>
            <p style="color: #94a3b8; margin: 0;">
                Isi data di sebelah kiri dan klik "Prediksi Sekarang" untuk melihat hasil prediksi tingkat stres Anda.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# BOTTOM SECTION: How It Works
# ============================================================================

st.divider()

st.markdown("""
<div style="text-align: center; margin: 40px 0 20px 0;">
    <h2 style="color: #f1f5f9;">🔍 Bagaimana Sistem Bekerja?</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5, gap="small")

steps = [
    ("📊\n**Input**", "Data dari 6 fitur mahasiswa"),
    ("🔀\n**Fuzzifikasi**", "Konversi ke membership fuzzy"),
    ("⚙️\n**Rule Base**", "Evaluasi 14+ aturan IF-THEN"),
    ("📈\n**Inferensi**", "Hitung firing strength"),
    ("🎯\n**Defuzzifikasi**", "Output: Rendah/Sedang/Tinggi"),
]

for idx, (col, (title, desc)) in enumerate(zip([col1, col2, col3, col4, col5], steps)):
    with col:
        st.markdown(f"""
        <div class="card" style="text-align: center; min-height: 150px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2em; margin-bottom: 10px;">{title}</div>
            <p style="font-size: 0.85em; color: #cbd5e1; margin: 0;">
                {desc}
            </p>
        </div>
        """, unsafe_allow_html=True)


st.divider()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.9em; padding: 20px 0;">
    <p style="margin: 5px 0;">
        🎓 Tugas Besar Kecerdasan Komputasional | Sistem Prediksi Tingkat Stres Mahasiswa
    </p>
    <p style="margin: 5px 0;">
        ⚠️ Sistem ini hanya alat bantu prediksi awal dan BUKAN diagnosis medis profesional.
    </p>
</div>
""", unsafe_allow_html=True)
