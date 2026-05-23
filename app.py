import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.predict import predict_stress
from src.recommendations import get_recommendation

st.set_page_config(
    page_title="Prediksi Tingkat Stres",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- UI/UX REDESIGN: GEN Z, SOFT DREAMY, GHIBLI AESTHETIC ---
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&display=swap');
    
    /* Global Background & Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Nunito', sans-serif !important;
        background: linear-gradient(135deg, #fdfbf7 0%, #f0f4ef 50%, #e6edf2 100%) !important; /* Soft creamy sky to subtle sage & blue */
        color: #4a5c50 !important; /* Dark warm olive text */
    }
    
    .stApp {
        background-attachment: fixed;
    }

    /* Hide standard top padding for cleaner look */
    .block-container {
        padding-top: 2rem !important;
        max-width: 1200px !important;
    }

    /* Glassmorphism Cards */
    .card {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 28px;
        box-shadow: 0 12px 40px rgba(92, 107, 92, 0.05);
        padding: 28px;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-2px);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(253, 251, 247, 0.85) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.6);
    }
    [data-testid="stSidebar"] * {
        color: #4a5c50 !important;
    }

    /* Pill Badges */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 16px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #5b7562;
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(138, 181, 148, 0.3);
        margin: 4px 6px 4px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
    }

    /* Primary Button Redesign */
    .stButton > button {
        background: linear-gradient(135deg, #8ab594 0%, #6d9677 100%) !important; /* Sage Green */
        color: #ffffff !important;
        border: none !important;
        border-radius: 999px !important; /* Pill shape */
        padding: 16px 28px !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px;
        box-shadow: 0 10px 25px rgba(109, 150, 119, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 30px rgba(109, 150, 119, 0.4) !important;
    }

    /* Progress Bar & Sliders */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #8ab594, #a4caba) !important;
        border-radius: 99px;
    }
    .stSlider > div > div > div {
        background: #8ab594 !important;
    }
    .stSlider > div > div > div > div {
        background: #ffffff !important;
        border: 2px solid #8ab594 !important;
        box-shadow: 0 2px 8px rgba(138, 181, 148, 0.4) !important;
    }

    /* Typography Overrides */
    h1, h2, h3, h4 {
        color: #3b4a3f !important;
        font-weight: 800 !important;
    }
    
    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 20px !important;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important;
    }
    div[data-testid="stExpander"] details summary p {
        font-weight: 700 !important;
        color: #4a5c50 !important;
        font-size: 1.05rem !important;
    }

    /* Metrics & Dataframes */
    .stMetric > div {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        border-radius: 24px !important;
        padding: 20px !important;
        box-shadow: 0 10px 30px rgba(92, 107, 92, 0.04) !important;
    }
    .stDataFrame table {
        border-radius: 16px;
        overflow: hidden;
    }
    
    /* Custom divider */
    hr {
        border-color: rgba(138, 181, 148, 0.2) !important;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3233/3233483.png", width=60) # Adding a cute subtle icon
    st.markdown("### 🌿 Tentang Sistem")
    st.write("Sistem prediksi awal tingkat stres mahasiswa dengan sentuhan visual yang menenangkan, ditenagai oleh Neuro-Fuzzy / ANFIS.")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Detail Proyek")
    st.markdown(
        """
        - **Nama:** Prediksi Tingkat Stres
        - **Mata Kuliah:** Kecerdasan Komputasional
        - **Dataset:** StressLevelDataset.csv
        - **Target:** Rendah / Sedang / Tinggi
        - **Engine:** Neuro-Fuzzy / ANFIS
        """
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.info("✨ Aplikasi ini adalah alat bantu prediksi awal, bukan diagnosis klinis.")


def get_factor_category(feature, value):
    if feature == "anxiety_level":
        if value <= 7: return "Rendah"
        if value <= 14: return "Sedang"
        return "Tinggi"
    if feature == "sleep_quality":
        if value <= 1: return "Buruk"
        if value <= 3: return "Sedang"
        return "Baik"
    if feature == "study_load":
        if value <= 1: return "Ringan"
        if value <= 3: return "Sedang"
        return "Berat"
    if feature == "academic_performance":
        if value <= 1: return "Rendah"
        if value <= 3: return "Sedang"
        return "Tinggi"
    if feature == "social_support":
        if value <= 1: return "Rendah"
        if value <= 3: return "Sedang"
        return "Tinggi"
    if feature == "future_career_concerns":
        if value <= 1: return "Rendah"
        if value <= 3: return "Sedang"
        return "Tinggi"
    return "Sedang"


def get_result_interpretation(label):
    if label == "Rendah":
        return "Kondisi stres relatif terkendali. Suasana hati secerah langit pagi. Pertahankan rutinitas baikmu! 🍃"
    if label == "Sedang":
        return "Ada sedikit awan mendung di pikiranmu. Coba kelola tekanan dengan lebih terstruktur dan ambil jeda sejenak. ☁️"
    return "Terdapat indikasi badai stres yang tinggi. Jangan ragu untuk mencari teman cerita atau konseling kampus ya. 🌧️"


def get_result_color(label):
    # Updated to Soft/Pastel Ghibli palette
    if label == "Rendah":
        return "#5b8064", "rgba(230, 239, 233, 0.8)", "#2a4030" # Soft Sage
    if label == "Sedang":
        return "#bd8d62", "rgba(253, 245, 239, 0.8)", "#5c3d24" # Warm Peach/Clay
    return "#bd6b6b", "rgba(250, 235, 235, 0.8)", "#5c2727" # Dusty Rose


def get_dominant_factors(input_data):
    metrics = [
        ("Kecemasan", input_data["anxiety_level"], input_data["anxiety_level"] / 20),
        ("Beban Belajar", input_data["study_load"], input_data["study_load"] / 5),
        ("Akademik", input_data["academic_performance"], (5 - input_data["academic_performance"]) / 5),
        ("Tidur", input_data["sleep_quality"], (5 - input_data["sleep_quality"]) / 5),
        ("Sosial", input_data["social_support"], (5 - input_data["social_support"]) / 5),
        ("Karier", input_data["future_career_concerns"], input_data["future_career_concerns"] / 5),
    ]
    ordered = sorted(metrics, key=lambda x: x[2], reverse=True)
    dominant = []
    for title, value, _ in ordered[:3]:
        if title == "Kecemasan":
            dominant.append((title, value, "Kecemasan yang tinggi ibarat kabut, sangat memengaruhi kondisimu."))
        elif title == "Beban Belajar":
            dominant.append((title, value, "Tumpukan tugas memberikan tekanan ekstra di pundakmu."))
        elif title == "Akademik":
            dominant.append((title, value, "Rasa kurang puas pada performa akademik cukup membebani pikiran."))
        elif title == "Tidur":
            dominant.append((title, value, "Kurang tidur membuat tubuhmu sulit memulihkan energi secara optimal."))
        elif title == "Sosial":
            dominant.append((title, value, "Kurangnya teman berbagi membuat beban terasa lebih berat dari biasanya."))
        else:
            dominant.append((title, value, "Terlalu memikirkan masa depan menyita banyak energi mentalmu saat ini."))
    return dominant


def render_stat_card(title, value, subtitle, color):
    st.markdown(
        f"""
        <div style='background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 24px; padding: 22px; min-height: 130px; box-shadow: 0 10px 30px rgba(0,0,0,0.02); transition: transform 0.3s ease;'>
            <div style='font-size: 0.95rem; font-weight: 700; color: #5b7562; margin-bottom: 8px;'>{title}</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: {color}; margin-bottom: 8px; line-height: 1.1;'>{value}</div>
            <div style='font-size: 0.85rem; color: #738276; font-weight: 500;'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- HERO SECTION ---
st.markdown(
    """
    <div style='background: linear-gradient(135deg, rgba(228, 238, 222, 0.5), rgba(230, 239, 245, 0.6)); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.9); border-radius: 32px; padding: 40px; box-shadow: 0 15px 40px rgba(138, 181, 148, 0.1); margin-bottom: 10px;'>
        <div style='max-width: 900px;'>
            <div style='font-size: 3rem; font-weight: 800; color: #3b4a3f; margin-bottom: 16px; letter-spacing: -0.5px; line-height: 1.2;'>
                Kenali Pikiranmu,<br>Kelola Stresmu. 🍃
            </div>
            <div style='font-size: 1.1rem; color: #5b7562; line-height: 1.7; margin-bottom: 24px; font-weight: 500;'>
                Prediksi awal tingkat stres berbasis Neuro-Fuzzy untuk bantu kamu memahami kondisi psikologis, akademik, dan sosial agar harimu lebih cerah.
            </div>
            <div style='display: flex; flex-wrap: wrap; gap: 8px;'>
                <span class='badge-pill'>✨ Neuro-Fuzzy</span>
                <span class='badge-pill'>☁️ ANFIS</span>
                <span class='badge-pill'>🌱 Mental Health</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
cols = st.columns(4)
with cols[0]: render_stat_card("Fitur Input", "6 Aspek", "Anxiety, Tidur, dll", "#6e967d")
with cols[1]: render_stat_card("Kelas Prediksi", "3 Level", "Rendah, Sedang, Tinggi", "#b07f85")
with cols[2]: render_stat_card("Output", "Personal", "Saran gaya hidup & akademik", "#bd8d62")
with cols[3]: render_stat_card("Dataset", "Kaggle", "Student Stress Factors", "#6c89a3")

st.write("")

col_input, col_result = st.columns([1.1, 1.3], gap="large")
with col_input:
    st.markdown(
        """
        <div style='margin-bottom: 20px; padding-left: 5px;'>
            <div style='font-size: 1.4rem; font-weight: 800; color: #3b4a3f; margin-bottom: 6px;'>Bagaimana kabarmu hari ini?</div>
            <p style='color: #6a7d6f; font-size: 0.95rem; line-height: 1.6; margin: 0;'>Geser slider di bawah dengan jujur untuk hasil terbaik.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("💭 Faktor Psikologis", expanded=True):
        anxiety_level = st.slider(
            "Tingkat Kecemasan", 0, 20, 10, 1,
            help="0 = setenang air danau, 20 = pikiran sangat bising.", format="%d"
        )

    with st.expander("📚 Faktor Akademik", expanded=True):
        study_load = st.slider(
            "Beban Belajar", 0, 5, 3, 1,
            help="0 = santai, 5 = tugas menumpuk bagai gunung.", format="%d"
        )
        academic_performance = st.slider(
            "Performa Akademik", 0, 5, 3, 1,
            help="0 = sedang underperform, 5 = sangat memuaskan.", format="%d"
        )

    with st.expander("🌙 Faktor Gaya Hidup", expanded=True):
        sleep_quality = st.slider(
            "Kualitas Tidur", 0, 5, 3, 1,
            help="0 = sering begadang/insomnia, 5 = tidur lelap.", format="%d"
        )

    with st.expander("🤝 Faktor Sosial & Karier", expanded=True):
        social_support = st.slider(
            "Dukungan Sosial", 0, 5, 3, 1,
            help="0 = merasa sendiri, 5 = dikelilingi orang tersayang.", format="%d"
        )
        future_career_concerns = st.slider(
            "Kekhawatiran Karier", 0, 5, 3, 1,
            help="0 = jalan terlihat jelas, 5 = overthinking masa depan.", format="%d"
        )

    st.write("")
    predict_button = st.button("✨ Cek Tingkat Stresku")

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
        label, score, class_scores = predict_stress(input_data)
        summary_text = get_recommendation(label)
        interpretation = get_result_interpretation(label)
        primary_color, pastel_color, text_color = get_result_color(label)

        # Result Card
        st.markdown(
            f"""
            <div style='background: {pastel_color}; backdrop-filter: blur(12px); border: 2px solid #ffffff; border-radius: 32px; padding: 32px; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.05); margin-bottom: 24px; animation: fadeIn 0.6s ease;'>
                <div style='font-size: 1.05rem; font-weight: 800; color: {text_color}; margin-bottom: 16px; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px;'>Hasil Prediksi</div>
                <div style='display: flex; gap: 24px; flex-wrap: wrap; align-items: center;'>
                    <div style='flex: 1 1 250px;'>
                        <div style='font-size: 2.8rem; font-weight: 800; color: {text_color}; margin-bottom: 8px; line-height: 1.1;'>{label}</div>
                        <div style='font-size: 1.05rem; color: {text_color}; line-height: 1.6; margin-bottom: 16px; opacity: 0.9;'>{interpretation}</div>
                    </div>
                    <div style='flex: 0 0 160px; background: rgba(255,255,255,0.6); padding: 20px; border-radius: 24px; text-align: center; box-shadow: inset 0 2px 10px rgba(255,255,255,0.5);'>
                        <div style='font-size: 2.2rem; font-weight: 800; color: {primary_color}; margin-bottom: 4px;'>{score:.0%}</div>
                        <div style='font-size: 0.85rem; color: {text_color}; font-weight: 700; opacity: 0.8;'>Akurasi Keyakinan</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Plot Customization (Softer Colors & Transparent BG)
        st.write("")
        fig, ax = plt.subplots(figsize=(7, 2.8))
        classes = list(class_scores.keys())
        values = list(class_scores.values())
        soft_colors = ["#8cb396", "#dfb48b", "#c77d7d"] # Soft Green, Warm Peach, Dusty Rose
        
        ax.bar(classes, values, color=soft_colors, edgecolor="white", linewidth=2.5, width=0.5)
        ax.set_ylim(0, 1)
        ax.set_title("Distribusi Probabilitas Kelas", fontsize=13, color="#4a5c50", pad=16, fontweight='bold', fontname='Nunito')
        
        # Make plot background transparent for clean UI
        ax.set_facecolor("none")
        fig.patch.set_facecolor("none")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(colors="#738276", bottom=False, left=False)
        ax.set_yticks([]) # Hide Y axis numbers for cleaner aesthetic
        
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.05, f"{value:.2f}", ha="center", va="bottom", color="#4a5c50", fontweight="800", fontname='Nunito')
        
        st.pyplot(fig)

        # Dominant Factors UI
        st.write("")
        st.markdown("<div style='font-size: 1.1rem; font-weight: 800; color: #3b4a3f; margin-bottom: 16px;'>🔍 Apa yang Paling Memengaruhimu?</div>", unsafe_allow_html=True)
        st.markdown("<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 20px; margin-bottom: 24px;'>", unsafe_allow_html=True)
        
        dominant = get_dominant_factors(input_data)
        for title, value, description in dominant:
            st.markdown(
                f"""
                <div style='background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.95); border-radius: 24px; padding: 22px; min-height: 170px; box-shadow: 0 10px 28px rgba(0,0,0,0.03);'>
                    <div style='font-size: 0.9rem; font-weight: 800; color: #8ab594; margin-bottom: 10px; text-transform: uppercase;'>{title}</div>
                    <div style='font-size: 1.8rem; font-weight: 800; color: #4a5c50; margin-bottom: 14px;'>{value} <span style='font-size: 0.9rem; color:#94a398;'>/ 5</span></div>
                    <div style='font-size: 0.9rem; color: #6a7d6f; line-height: 1.7; font-weight: 500;'>{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Recommendation Section
        st.write("")
        if label == "Rendah":
            reco_cards = [
                ("📚 Akademik", "Pertahankan jadwal belajar teratur dan hindari penundaan tugas."),
                ("🍃 Gaya Hidup", "Jaga pola tidur yang baik dan sempatkan jalan santai menghirup udara segar."),
                ("🤝 Sosial", "Bagikan energi positifmu dengan teman-teman di sekitar.")
            ]
        elif label == "Sedang":
            reco_cards = [
                ("📚 Akademik", "Coba pecah tugas besar menjadi langkah-langkah kecil agar tidak kewalahan."),
                ("🍃 Gaya Hidup", "Seduh teh hangat, dengarkan musik lo-fi, dan pastikan istirahatmu cukup."),
                ("🤝 Sosial", "Luangkan waktu mengobrol dengan orang yang membuatmu merasa aman."),
            ]
        else:
            reco_cards = [
                ("📚 Akademik", "Kurangi ekspektasi berlebih untuk sementara. Minta bantuan dosen/teman jika buntu."),
                ("🍃 Gaya Hidup", "Tarik napas panjang. Kurangi kafein dan prioritaskan tidur yang berkualitas."),
                ("🤝 Sosial", "Pikiranmu sedang penuh. Jangan ragu membagi beban dengan konselor atau sahabat."),
            ]

        st.markdown(
            """
            <div style='background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(15px); border: 1px solid rgba(255,255,255,0.9); border-radius: 30px; padding: 32px; margin-top: 24px; box-shadow: 0 15px 35px rgba(0,0,0,0.03);'>
                <div style='font-size: 1.3rem; font-weight: 800; color: #3b4a3f; margin-bottom: 24px;'>💌 Catatan Untukmu</div>
            """,
            unsafe_allow_html=True,
        )
        for title, text in reco_cards:
            st.markdown(
                f"""
                <div style='margin-bottom: 20px; background: rgba(255,255,255,0.5); padding: 16px 20px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.8);'>
                    <div style='font-size: 1rem; font-weight: 800; color: #5b7562; margin-bottom: 6px;'>{title}</div>
                    <div style='font-size: 0.95rem; color: #5c6b60; line-height: 1.6; font-weight: 500;'>{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Empty State
        st.markdown(
            """
            <div style='background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(15px); border: 1px solid rgba(255,255,255,0.9); border-radius: 32px; padding: 40px; box-shadow: 0 15px 40px rgba(0, 0, 0, 0.03); text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 400px;'>
                <div style='font-size: 4rem; margin-bottom: 16px;'>🌱</div>
                <div style='font-size: 1.5rem; font-weight: 800; color: #3b4a3f; margin-bottom: 12px;'>Ruang Aman Untukmu</div>
                <p style='color: #6a7d6f; font-size: 1.05rem; line-height: 1.7; margin-bottom: 32px; max-width: 400px; font-weight: 500;'>Isi indikator di sebelah kiri dan klik tombol prediksi untuk melihat apa yang sedang terjadi di pikiranmu saat ini.</p>
                <div style='display: flex; gap: 16px; flex-wrap: wrap; justify-content: center;'>
                    <div style='background: rgba(138, 181, 148, 0.15); border-radius: 20px; padding: 12px 24px; color: #4c5b50; font-weight: 700; font-size: 0.9rem;'>💭 Analisis Emosi</div>
                    <div style='background: rgba(223, 180, 139, 0.15); border-radius: 20px; padding: 12px 24px; color: #8c6a47; font-weight: 700; font-size: 0.9rem;'>🎯 Saran Personal</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Footer Information
st.write("")
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; margin-top: 10px;'>
        <div style='background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.9); border-radius: 28px; padding: 28px;'>
            <div style='font-size: 1.15rem; font-weight: 800; color: #3b4a3f; margin-bottom: 16px;'>⚙️ Cara Kerja Sistem</div>
            <p style='color: #6a7d6f; font-size: 0.95rem; line-height: 1.7; font-weight: 500;'>Metode Neuro-Fuzzy (ANFIS) bekerja ibarat otak manusia. Ia mengubah angka kaku menjadi linguistik lembut (Fuzzifikasi), menimbang aturan (Rule), dan belajar dari pola (Neural Network) untuk memberikan prediksi yang adaptif.</p>
        </div>
        <div style='background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.9); border-radius: 28px; padding: 28px;'>
            <div style='font-size: 1.15rem; font-weight: 800; color: #3b4a3f; margin-bottom: 16px;'>📌 Catatan Akademik</div>
            <p style='color: #6a7d6f; font-size: 0.95rem; line-height: 1.7; font-weight: 500;'>Sistem menggunakan FIS sebagai kerangka logis dan ANFIS sebagai mesin belajar. Aplikasi ini dirancang untuk ranah komputasional dan kesadaran kesehatan mental dini, bukan rujukan medis.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)