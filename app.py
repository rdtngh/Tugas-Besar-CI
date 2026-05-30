import streamlit as st
import matplotlib.pyplot as plt
from src.predict import predict_stress
from src.analysis import (
    classify_input_level,
    get_factor_contribution,
    generate_analysis_text,
    generate_dynamic_recommendations,
)

st.set_page_config(
    page_title="Prediksi Tingkat Stres Mahasiswa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ACADEMIC CLEAN DESIGN WITH SOFT COLOR PALETTE ---
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Background & Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background: #F7FAF8 !important;
        color: #263238 !important;
    }
    
    .stApp {
        background-attachment: fixed;
    }

    /* Block Container */
    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1400px !important;
    }

    /* Sidebar Styling - Academic Light */
    [data-testid="stSidebar"] {
        background: #F7FAF8 !important;
        border-right: 1px solid #E5E7EB !important;
    }
    [data-testid="stSidebar"] * {
        color: #263238 !important;
    }

    /* Primary Button - Academic Green */
    .stButton > button {
        background: #2F5D50 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(47, 93, 80, 0.15) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: #254A41 !important;
        box-shadow: 0 4px 12px rgba(47, 93, 80, 0.25) !important;
    }

    /* Sliders - Clean Academic Style with Teal Color */
    .stSlider > div > div > div {
        background: #7BAE9D !important;
    }
    .stSlider > div > div > div > div {
        background: #FFFFFF !important;
        border: 2px solid #7BAE9D !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1) !important;
    }

    /* Slider Label - Clear and Prominent */
    .stSlider > label {
        font-weight: 600 !important;
        color: #263238 !important;
        font-size: 0.95rem !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #263238 !important;
        font-weight: 700 !important;
    }
    
    /* Expanders - Clean Cards with Dark Headers */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        overflow: hidden !important;
    }
    
    /* Expander Header - Dark Background with Light Text */
    div[data-testid="stExpander"] details summary {
        background: #1F2937 !important;
        padding: 16px !important;
        border-radius: 12px 12px 0 0 !important;
    }
    
    div[data-testid="stExpander"] details summary p {
        font-weight: 700 !important;
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
    }

    /* Expander Content Padding */
    div[data-testid="stExpander"] details {
        padding: 0 !important;
    }
    
    div[data-testid="stExpander"] details > div {
        padding: 20px !important;
    }

    /* Metrics & Cards */
    .stMetric > div {
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    .stDataFrame table {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Divider */
    hr {
        border-color: #E5E7EB !important;
    }

    /* Info/Alert boxes */
    .stAlert > div {
        background: #F7FAF8 !important;
        border-left: 4px solid #2F5D50 !important;
    }

    /* Custom Slider Label Styles */
    .slider-label {
        font-weight: 600 !important;
        color: #263238 !important;
        font-size: 0.95rem !important;
        display: block;
        margin-bottom: 4px;
    }

    .slider-scale {
        font-size: 0.8rem !important;
        color: #6B7280 !important;
        display: block;
        margin-bottom: 10px;
        font-style: italic;
    }

    .slider-value {
        font-size: 0.8rem !important;
        color: #7BAE9D !important;
        display: block;
        margin-top: 8px;
        font-weight: 600;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("### Tentang Sistem")
    st.markdown(
        """
        Sistem prediksi awal tingkat stres mahasiswa menggunakan pendekatan **simplified Neuro-Fuzzy/ANFIS** sebagai model utama dan **Fuzzy Inference System (FIS)** sebagai fallback.
        """
    )
    
    st.markdown("---")
    
    st.markdown("#### Informasi Project")
    st.markdown(
        """
        - **Mata Kuliah:** Kecerdasan Komputasional
        - **Semester:** 6
        - **Dataset:** StressLevelDataset.csv (Kaggle)
        - **Kelas Output:** Rendah, Sedang, Tinggi
        """
    )
    
    st.markdown("---")
    
    st.markdown("#### Dataset")
    st.markdown("**6 fitur input:**")
    st.markdown(
        """
        - Anxiety Level (0-20)
        - Sleep Quality (0-5)
        - Study Load (0-5)
        - Academic Performance (0-5)
        - Social Support (0-5)
        - Future Career Concerns (0-5)
        """
    )
    
    st.markdown("---")
    
    st.markdown("#### Metode")
    st.markdown(
        """
        **Simplified ANFIS:**
        - Membership Gaussian adaptif
        - Rule layer dengan firing strength
        - Consequent layer linear + Softmax
        - Training dengan gradient descent
        
        **FIS Fallback:**
        - Membership triangular/trapezoidal statis
        - Rule base berbasis IF-THEN
        """
    )
    
    st.markdown("---")
    
    st.warning("⚠️ Sistem ini adalah alat bantu prediksi awal, bukan diagnosis medis. Untuk konsultasi profesional, hubungi layanan kesehatan mental kampus.")



def get_result_interpretation(label):
    """Get interpretation text based on stress level."""
    if label == "Rendah":
        return "Kondisi stres Anda relatif terkendali. Pertahankan gaya hidup sehat dan rutinitas baik yang sudah berjalan."
    if label == "Sedang":
        return "Anda mengalami tingkat stres sedang. Pertimbangkan untuk lebih mengelola workload dan mengambil waktu istirahat yang cukup."
    return "Tingkat stres Anda tinggi. Sangat disarankan untuk mencari dukungan dari konselor, teman dekat, atau layanan kesehatan mental kampus."


def get_result_color(label):
    """Get colors for result card based on stress level."""
    if label == "Rendah":
        return "#059669", "rgba(220, 252, 231, 0.6)", "#047857"  # Green
    if label == "Sedang":
        return "#F97316", "rgba(254, 243, 230, 0.6)", "#EA580C"  # Orange
    return "#DC2626", "rgba(254, 226, 226, 0.6)", "#991B1B"  # Red


def render_info_card(title, value, description, bg_color, text_color):
    """Render small info card."""
    st.markdown(
        f"""
        <div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 18px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
            <div style='font-size: 0.85rem; font-weight: 600; color: #6B7280; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;'>{title}</div>
            <div style='font-size: 1.5rem; font-weight: 700; color: {text_color}; margin-bottom: 6px;'>{value}</div>
            <div style='font-size: 0.8rem; color: #6B7280; line-height: 1.4;'>{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- HEADER SECTION ---
st.markdown(
    """
    <div style='margin-bottom: 32px;'>
        <h1 style='color: #263238; margin-bottom: 8px;'>Sistem Prediksi Tingkat Stres Mahasiswa</h1>
        <p style='color: #6B7280; font-size: 1rem; line-height: 1.6; margin: 0;'>
            Menggunakan pendekatan <strong>simplified Neuro-Fuzzy/ANFIS</strong> dan <strong>Fuzzy Inference System</strong> 
            untuk prediksi awal tingkat stres mahasiswa berdasarkan indikator psikologis, akademik, dan sosial.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- INFO CARDS ---
col1, col2, col3, col4 = st.columns(4, gap="medium")
with col1:
    render_info_card("Fitur Input", "6", "Aspek psikologis & akademik", "#E0F2FE", "#0369A1")
with col2:
    render_info_card("Kelas Output", "3", "Rendah, Sedang, Tinggi", "#DCFCE7", "#166534")
with col3:
    render_info_card("Model Utama", "ANFIS", "Simplified Neuro-Fuzzy", "#FEF3C7", "#92400E")
with col4:
    render_info_card("Fallback", "FIS", "Fuzzy Inference System", "#FCE7F3", "#831843")



st.markdown("---")

# --- MAIN CONTENT AREA ---
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("#### Form Input Prediksi")
    st.markdown("Masukkan nilai indikator di bawah, lalu klik tombol prediksi untuk melihat hasil.")
    st.markdown("")

    # --- FAKTOR PSIKOLOGIS ---
    with st.expander("Faktor Psikologis", expanded=True):
        st.markdown('<div class="slider-label">Tingkat Kecemasan</div>', unsafe_allow_html=True)
        st.markdown('<div class="slider-scale">Skala: 0 = sangat rendah, 20 = sangat tinggi</div>', unsafe_allow_html=True)
        anxiety_level = st.slider(
            label="Tingkat Kecemasan",
            min_value=0,
            max_value=20,
            value=10,
            step=1,
            label_visibility="collapsed",
            help="0 = sangat tenang | 20 = sangat cemas"
        )
        st.markdown(f'<div class="slider-value">Skor: {anxiety_level} / 20</div>', unsafe_allow_html=True)
        st.markdown("")

    # --- FAKTOR AKADEMIK ---
    with st.expander("Faktor Akademik", expanded=True):
        st.markdown('<div class="slider-label">Beban Belajar</div>', unsafe_allow_html=True)
        st.markdown('<div class="slider-scale">Skala: 0 = sangat ringan, 5 = sangat berat</div>', unsafe_allow_html=True)
        study_load = st.slider(
            label="Beban Belajar",
            min_value=0,
            max_value=5,
            value=3,
            step=1,
            label_visibility="collapsed",
            help="0 = tidak ada beban | 5 = beban sangat tinggi"
        )
        st.markdown(f'<div class="slider-value">Skor: {study_load} / 5</div>', unsafe_allow_html=True)
        st.markdown("")
        
        st.markdown('<div class="slider-label">Performa Akademik</div>', unsafe_allow_html=True)
        st.markdown('<div class="slider-scale">Skala: 0 = sangat rendah, 5 = sangat baik</div>', unsafe_allow_html=True)
        academic_performance = st.slider(
            label="Performa Akademik",
            min_value=0,
            max_value=5,
            value=3,
            step=1,
            label_visibility="collapsed",
            help="0 = sangat kurang memuaskan | 5 = sangat memuaskan"
        )
        st.markdown(f'<div class="slider-value">Skor: {academic_performance} / 5</div>', unsafe_allow_html=True)
        st.markdown("")

    # --- FAKTOR GAYA HIDUP ---
    with st.expander("Faktor Gaya Hidup", expanded=True):
        st.markdown('<div class="slider-label">Kualitas Tidur</div>', unsafe_allow_html=True)
        st.markdown('<div class="slider-scale">Skala: 0 = sangat buruk, 5 = sangat baik</div>', unsafe_allow_html=True)
        sleep_quality = st.slider(
            label="Kualitas Tidur",
            min_value=0,
            max_value=5,
            value=3,
            step=1,
            label_visibility="collapsed",
            help="0 = sering begadang/insomnia | 5 = tidur lelap"
        )
        st.markdown(f'<div class="slider-value">Skor: {sleep_quality} / 5</div>', unsafe_allow_html=True)
        st.markdown("")

    # --- FAKTOR SOSIAL DAN KARIER ---
    with st.expander("Faktor Sosial dan Karier", expanded=True):
        st.markdown('<div class="slider-label">Dukungan Sosial</div>', unsafe_allow_html=True)
        st.markdown('<div class="slider-scale">Skala: 0 = sangat rendah, 5 = sangat tinggi</div>', unsafe_allow_html=True)
        social_support = st.slider(
            label="Dukungan Sosial",
            min_value=0,
            max_value=5,
            value=3,
            step=1,
            label_visibility="collapsed",
            help="0 = merasa sendiri | 5 = dikelilingi orang tersayang"
        )
        st.markdown(f'<div class="slider-value">Skor: {social_support} / 5</div>', unsafe_allow_html=True)
        st.markdown("")
        
        st.markdown('<div class="slider-label">Kekhawatiran Karier</div>', unsafe_allow_html=True)
        st.markdown('<div class="slider-scale">Skala: 0 = sangat rendah, 5 = sangat tinggi</div>', unsafe_allow_html=True)
        future_career_concerns = st.slider(
            label="Kekhawatiran Karier",
            min_value=0,
            max_value=5,
            value=3,
            step=1,
            label_visibility="collapsed",
            help="0 = tidak khawatir | 5 = sangat khawatir"
        )
        st.markdown(f'<div class="slider-value">Skor: {future_career_concerns} / 5</div>', unsafe_allow_html=True)
        st.markdown("")

    st.markdown("")
    predict_button = st.button("Jalankan Prediksi", use_container_width=True)

with col_result:
    if predict_button:
        input_data = {
            "anxiety_level": anxiety_level,
            "sleep_quality": sleep_quality,
            "study_load": study_load,
            "academic_performance": academic_performance,
            "social_support": social_support,
            "future_career_concerns": future_career_concerns,
        }
        
        label, score, class_scores, model_used = predict_stress(input_data)
        interpretation = get_result_interpretation(label)
        primary_color, pastel_bg, text_color = get_result_color(label)
        
        # Result Card
        st.markdown(
            f"""
            <div style='background: {pastel_bg}; border: 2px solid #E5E7EB; border-radius: 12px; padding: 28px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);'>
                <div style='font-size: 0.85rem; font-weight: 700; color: {text_color}; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;'>Hasil Prediksi</div>
                <div style='font-size: 2.5rem; font-weight: 700; color: {text_color}; margin-bottom: 12px; line-height: 1.1;'>{label}</div>
                <div style='font-size: 0.95rem; color: {text_color}; line-height: 1.6; margin-bottom: 16px;'>{interpretation}</div>
                <div style='background: rgba(255,255,255,0.7); border: 1px solid #E5E7EB; padding: 12px 16px; border-radius: 8px; display: inline-block; font-weight: 600; color: {text_color};'>
                    Confidence Score: <strong>{score:.0%}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model & Class Scores Info
        st.markdown("")
        col_model, col_scores = st.columns(2)
        
        with col_model:
            model_badge = "Simplified ANFIS" if model_used == "Simplified ANFIS" else "FIS (Fallback)"
            st.markdown(
                f"""
                <div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 14px; text-align: center;'>
                    <div style='font-size: 0.75rem; color: #6B7280; margin-bottom: 6px; text-transform: uppercase;'>Model Digunakan</div>
                    <div style='font-size: 0.95rem; font-weight: 700; color: #263238;'>{model_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col_scores:
            st.markdown(
                f"""
                <div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 14px; text-align: center;'>
                    <div style='font-size: 0.75rem; color: #6B7280; margin-bottom: 6px; text-transform: uppercase;'>Kelas Terprediksi</div>
                    <div style='font-size: 0.95rem; font-weight: 700; color: #263238;'>{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Class Scores Distribution
        st.markdown("")
        st.markdown("**Distribusi Probabilitas Kelas**")
        
        fig, ax = plt.subplots(figsize=(7, 2.5))
        classes = list(class_scores.keys())
        values = list(class_scores.values())
        colors = ["#059669", "#F97316", "#DC2626"]  # Green, Orange, Red
        
        bars = ax.bar(classes, values, color=colors, edgecolor="#E5E7EB", linewidth=1.5, width=0.5, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_title("", fontsize=0)
        
        ax.set_facecolor("none")
        fig.patch.set_facecolor("none")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#E5E7EB")
        ax.tick_params(colors="#6B7280", bottom=False, left=True)
        ax.set_yticklabels([f"{i:.1f}" for i in ax.get_yticks()], fontsize=9)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                   f"{value:.3f}", ha="center", va="bottom", fontweight="600", fontsize=10, color="#263238")
        
        st.pyplot(fig, use_container_width=False)

        # Detail Input Values
        st.markdown("")
        st.markdown("**Detail Nilai Input**")
        
        input_table = {
            "Indikator": ["Kecemasan", "Beban Belajar", "Performa Akademik", "Kualitas Tidur", "Dukungan Sosial", "Kekhawatiran Karier"],
            "Nilai": [
                f"{anxiety_level}/20",
                f"{study_load}/5",
                f"{academic_performance}/5",
                f"{sleep_quality}/5",
                f"{social_support}/5",
                f"{future_career_concerns}/5",
            ]
        }
        
        st.markdown("")
        st.markdown("### Analisis Sistem")

        analysis_text = generate_analysis_text(input_data, label, class_scores)
        st.markdown(
            f"""
            <div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 22px; margin-bottom: 18px;'>
                <div style='font-size: 0.95rem; color: #263238; line-height: 1.7;'>{analysis_text}</div>
                <div style='font-size: 0.82rem; color: #6B7280; margin-top: 14px;'>Analisis ini dibuat berdasarkan nilai input pengguna dan hasil prediksi model. Analisis ini bukan diagnosis medis.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        contributions = get_factor_contribution(input_data)
        recommendations = generate_dynamic_recommendations(input_data, label)

        factor_cards = []
        for factor in contributions:
            factor_cards.append(
                f"""
                <div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 16px; margin-bottom: 12px;'>
                    <div style='font-size: 0.95rem; font-weight: 700; color: #263238; margin-bottom: 6px;'>{factor['factor']} — {factor['level'].capitalize()}</div>
                    <div style='font-size: 0.9rem; color: #6B7280; line-height: 1.6;'>{factor['reason']}</div>
                </div>
                """
            )

        recommendation_items = []
        for recommendation in recommendations:
            recommendation_items.append(
                f"<li style='margin-bottom: 10px; line-height: 1.6; color: #334155;'>{recommendation}</li>"
            )

        st.markdown("**Faktor yang Mempengaruhi Prediksi**")
        for card in factor_cards:
            st.markdown(card, unsafe_allow_html=True)

        st.markdown("**Rekomendasi Personal**")
        st.markdown(
            f"""
            <div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 10px; padding: 16px; margin-bottom: 18px;'>
                <ul style='margin: 0; padding-left: 20px;'>{''.join(recommendation_items)}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        # Empty State
        st.markdown(
            """
            <div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 40px; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 500px;'>
                <div style='font-size: 3rem; margin-bottom: 16px;'>📊</div>
                <div style='font-size: 1.3rem; font-weight: 700; color: #263238; margin-bottom: 10px;'>Belum Ada Prediksi</div>
                <p style='color: #6B7280; font-size: 0.95rem; line-height: 1.6; margin-bottom: 20px; max-width: 350px;'>
                    Masukkan nilai indikator pada form di sebelah kiri, kemudian klik tombol prediksi untuk melihat hasil analisis tingkat stres Anda.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )



# --- FOOTER SECTION ---
st.markdown("---")

st.markdown("### Cara Kerja Sistem")
st.markdown(
    """
    <div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 20px; margin-bottom: 18px;'>
        <p style='font-size: 0.95rem; color: #263238; line-height: 1.7; margin: 0;'>
        Dataset diproses melalui tahap preprocessing, kemudian digunakan oleh model simplified ANFIS untuk menghasilkan prediksi tingkat stres. Jika model utama tidak tersedia, sistem menggunakan FIS berbasis aturan sebagai fallback. Hasil prediksi ditampilkan bersama confidence score, faktor yang memengaruhi, dan rekomendasi personal.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Catatan Akademik")
st.markdown(
    """
    <div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 20px; margin-bottom: 18px;'>
        <p style='font-size: 0.95rem; color: #263238; line-height: 1.7; margin: 0;'>
        Sistem ini dibuat untuk keperluan tugas besar Kecerdasan Komputasional. Hasil prediksi bersifat informasi awal dan bukan diagnosis medis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

footer_html = """
<div style='background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: 24px; text-align: center; margin-top: 16px;'>
    <div style='color: #2F5D50; font-size: 1.05rem; font-weight: 700; margin-bottom: 8px;'>Sistem Prediksi Tingkat Stres Mahasiswa</div>
    <div style='color: #374151; font-size: 0.95rem; margin-bottom: 6px;'>Dibuat oleh Kelompok Third Chance</div>
    <div style='color: #6B7280; font-size: 0.9rem; margin-bottom: 12px;'>Tugas Besar Kecerdasan Komputasional • Semester 6</div>
    <div style='color: #6B7280; font-size: 0.85rem;'>Institut Teknologi Sumatera • 2026</div>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)