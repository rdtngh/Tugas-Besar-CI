import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.predict import predict_stress
from src.recommendations import get_recommendation

st.set_page_config(
    page_title="Sistem Prediksi Tingkat Stres Mahasiswa",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background: radial-gradient(circle at top center, rgba(56, 189, 248, 0.12), transparent 35%), #f8fafc !important;
        color: #0f172a !important;
    }
    .stApp {
        padding-top: 16px;
    }
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.05);
        padding: 24px;
        margin-bottom: 20px;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #0f172a;
        background: rgba(59, 130, 246, 0.12);
        margin: 4px 6px 4px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 14px 22px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 16px 36px rgba(37, 99, 235, 0.18) !important;
        transition: transform 0.18s ease, opacity 0.18s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        opacity: 0.95 !important;
    }
    .stProgress > div > div > div {
        background: #2563eb !important;
    }
    .stSlider > div > div > div {
        background: #2563eb !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #0f172a;
    }
    .stMetric > div {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 18px !important;
        padding: 20px !important;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.04) !important;
    }
    .stDataFrame table {
        border-color: #e2e8f0 !important;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Tentang Sistem")
    st.write("Sistem prediksi awal tingkat stres mahasiswa menggunakan pendekatan Neuro-Fuzzy / ANFIS.")
    st.markdown("---")
    st.subheader("Detail Proyek")
    st.markdown(
        """
        - **Nama Proyek:** Sistem Prediksi Tingkat Stres Mahasiswa
        - **Mata Kuliah:** Kecerdasan Komputasional
        - **Dataset:** StressLevelDataset.csv
        - **Target:** stress_level *(Rendah/Sedang/Tinggi)*
        - **Metode Utama:** Neuro-Fuzzy / ANFIS sederhana
        - **Baseline/Fallback:** Fuzzy Inference System
        """
    )
    st.markdown("---")
    st.warning("Sistem ini adalah alat bantu prediksi awal dan bukan diagnosis medis.")


def get_factor_category(feature, value):
    if feature == "anxiety_level":
        if value <= 7:
            return "Rendah"
        if value <= 14:
            return "Sedang"
        return "Tinggi"
    if feature == "sleep_quality":
        if value <= 1:
            return "Buruk"
        if value <= 3:
            return "Sedang"
        return "Baik"
    if feature == "study_load":
        if value <= 1:
            return "Ringan"
        if value <= 3:
            return "Sedang"
        return "Berat"
    if feature == "academic_performance":
        if value <= 1:
            return "Rendah"
        if value <= 3:
            return "Sedang"
        return "Tinggi"
    if feature == "social_support":
        if value <= 1:
            return "Rendah"
        if value <= 3:
            return "Sedang"
        return "Tinggi"
    if feature == "future_career_concerns":
        if value <= 1:
            return "Rendah"
        if value <= 3:
            return "Sedang"
        return "Tinggi"
    return "Sedang"


def get_result_interpretation(label):
    if label == "Rendah":
        return "Kondisi stres relatif terkendali. Pertahankan kebiasaan baik dan stabilkan ritme studi."
    if label == "Sedang":
        return "Terdapat indikasi tekanan yang perlu diperhatikan dan dikelola dengan lebih terstruktur."
    return "Terdapat indikasi stres tinggi, disarankan mencari dukungan sosial atau konseling kampus."


def get_result_color(label):
    if label == "Rendah":
        return "#0f766e", "#d1fae5", "#134e4a"
    if label == "Sedang":
        return "#b45309", "#fef3c7", "#92400e"
    return "#b91c1c", "#fee2e2", "#7f1d1d"


def get_dominant_factors(input_data):
    metrics = [
        ("Tingkat Kecemasan", input_data["anxiety_level"], input_data["anxiety_level"] / 20),
        ("Beban Belajar", input_data["study_load"], input_data["study_load"] / 5),
        ("Performa Akademik", input_data["academic_performance"], (5 - input_data["academic_performance"]) / 5),
        ("Kualitas Tidur", input_data["sleep_quality"], (5 - input_data["sleep_quality"]) / 5),
        ("Dukungan Sosial", input_data["social_support"], (5 - input_data["social_support"]) / 5),
        ("Kekhawatiran Karier", input_data["future_career_concerns"], input_data["future_career_concerns"] / 5),
    ]
    ordered = sorted(metrics, key=lambda x: x[2], reverse=True)
    dominant = []
    for title, value, _ in ordered[:3]:
        if title == "Tingkat Kecemasan":
            dominant.append((title, value, "Kecemasan tinggi berperan besar dalam prediksi stres."))
        elif title == "Beban Belajar":
            dominant.append((title, value, "Beban belajar tinggi dapat memperkuat tekanan akademik."))
        elif title == "Performa Akademik":
            dominant.append((title, value, "Performa akademik rendah cenderung meningkatkan risiko stres."))
        elif title == "Kualitas Tidur":
            dominant.append((title, value, "Tidur kurang berkualitas memengaruhi kemampuan pulih dari tekanan."))
        elif title == "Dukungan Sosial":
            dominant.append((title, value, "Kurangnya dukungan sosial membuat tekanan lebih sulit dikelola."))
        else:
            dominant.append((title, value, "Kekhawatiran karier yang tinggi menambah beban mental."))
    return dominant


def render_stat_card(title, value, subtitle, color):
    st.markdown(
        f"""
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 20px; padding: 20px; min-height: 120px;'>
            <div style='font-size: 0.93rem; color: #475569; margin-bottom: 10px;'>{title}</div>
            <div style='font-size: 1.7rem; font-weight: 700; color: {color}; margin-bottom: 8px;'>{value}</div>
            <div style='font-size: 0.88rem; color: #64748b;'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(title, description, color):
    st.markdown(
        f"""
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 22px; padding: 18px; min-height: 120px;'>
            <div style='font-size: 0.98rem; font-weight: 700; color: {color}; margin-bottom: 10px;'>{title}</div>
            <div style='font-size: 0.9rem; color: #475569; line-height: 1.7;'>{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style='background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(124,58,237,0.12)); border-radius: 30px; padding: 32px;'>
        <div style='max-width: 900px;'>
            <div style='font-size: 2.7rem; font-weight: 800; color: #0f172a; margin-bottom: 12px;'>Sistem Prediksi Tingkat Stres Mahasiswa</div>
            <div style='font-size: 1.05rem; color: #475569; line-height: 1.8; margin-bottom: 20px;'>Prediksi awal tingkat stres berbasis Neuro-Fuzzy untuk membantu memahami kondisi psikologis, akademik, sosial, dan karier mahasiswa.</div>
            <div style='display: flex; flex-wrap: wrap; gap: 10px;'>
                <span class='badge-pill'>Neuro-Fuzzy</span>
                <span class='badge-pill'>ANFIS</span>
                <span class='badge-pill'>Student Stress</span>
                <span class='badge-pill'>Computational Intelligence</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
cols = st.columns(4)
render_stat_card("Fitur Input", "6", "Anxiety, Tidur, Belajar, Akademik, Sosial, Karier", "#2563eb")
render_stat_card("Kelas Prediksi", "3", "Rendah, Sedang, Tinggi", "#7c3aed")
render_stat_card("Output", "Rekomendasi", "Saran personal sesuai level stres", "#06b6d4")
render_stat_card("Dataset", "Student Stress Factors", "Dataset penelitian akademik", "#2563eb")

st.write("")

col_input, col_result = st.columns([1.15, 1.25], gap="large")
with col_input:
    st.markdown(
        """
        <div class='card'>
            <div style='font-size: 1.2rem; font-weight: 700; color: #0f172a; margin-bottom: 10px;'>Masukkan Data Mahasiswa</div>
            <p style='color: #475569; line-height: 1.7; margin: 0;'>Isi 6 fitur untuk mendapatkan prediksi yang akurat dan rekomendasi yang sesuai.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Faktor Psikologis", expanded=True):
        st.markdown("**Tingkat Kecemasan**")
        anxiety_level = st.slider(
            "Tingkat Kecemasan",
            min_value=0,
            max_value=20,
            value=10,
            step=1,
            help="0 = sangat tenang, 20 = sangat cemas.",
            format="%d",
        )
        st.caption(f"Nilai kecemasan mahasiswa: {anxiety_level} / 20")

    with st.expander("Faktor Akademik", expanded=True):
        st.markdown("**Beban Belajar**")
        study_load = st.slider(
            "Beban Belajar",
            min_value=0,
            max_value=5,
            value=3,
            step=1,
            help="0 = sangat ringan, 5 = sangat berat.",
            format="%d",
        )
        st.caption(f"Nilai beban belajar: {study_load} / 5")

        st.markdown("**Performa Akademik**")
        academic_performance = st.slider(
            "Performa Akademik",
            min_value=0,
            max_value=5,
            value=3,
            step=1,
            help="0 = sangat rendah, 5 = sangat tinggi.",
            format="%d",
        )
        st.caption(f"Nilai performa akademik: {academic_performance} / 5")

    with st.expander("Faktor Gaya Hidup", expanded=True):
        st.markdown("**Kualitas Tidur**")
        sleep_quality = st.slider(
            "Kualitas Tidur",
            min_value=0,
            max_value=5,
            value=3,
            step=1,
            help="0 = sangat buruk, 5 = sangat baik.",
            format="%d",
        )
        st.caption(f"Nilai kualitas tidur: {sleep_quality} / 5")

    with st.expander("Faktor Sosial dan Karier", expanded=True):
        st.markdown("**Dukungan Sosial**")
        social_support = st.slider(
            "Dukungan Sosial",
            min_value=0,
            max_value=5,
            value=3,
            step=1,
            help="0 = sangat minim, 5 = sangat kuat.",
            format="%d",
        )
        st.caption(f"Nilai dukungan sosial: {social_support} / 5")

        st.markdown("**Kekhawatiran Karier**")
        future_career_concerns = st.slider(
            "Kekhawatiran Karier",
            min_value=0,
            max_value=5,
            value=3,
            step=1,
            help="0 = tidak khawatir, 5 = sangat khawatir.",
            format="%d",
        )
        st.caption(f"Nilai kekhawatiran karier: {future_career_concerns} / 5")

    st.write("")
    predict_button = st.button("Prediksi Sekarang")

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

        st.markdown(
            f"""
            <div style='background: {pastel_color}; border: 1px solid {primary_color}; border-radius: 26px; padding: 28px; box-shadow: 0 25px 55px rgba(15, 23, 42, 0.08);'>
                <div style='font-size: 1rem; font-weight: 700; color: {text_color}; margin-bottom: 12px;'>Hasil Prediksi</div>
                <div style='display: flex; gap: 22px; flex-wrap: wrap;'>
                    <div style='flex: 1 1 280px;'>
                        <div style='font-size: 2.3rem; font-weight: 800; color: #0f172a; margin-bottom: 10px;'>{label}</div>
                        <div style='font-size: 0.96rem; color: #334155; line-height: 1.75; margin-bottom: 12px;'>{interpretation}</div>
                        <div style='font-size: 0.88rem; color: #475569;'>Hasil ini merupakan prediksi awal, bukan diagnosis medis.</div>
                    </div>
                    <div style='flex: 0 0 180px; text-align: right;'>
                        <div style='font-size: 2.1rem; font-weight: 800; color: {primary_color}; margin-bottom: 4px;'>{score:.0%}</div>
                        <div style='font-size: 0.95rem; color: #475569;'>Confidence Score</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(score)
        st.caption("Confidence score berdasarkan output Neuro-Fuzzy. Nilai mendekati 1.00 menunjukkan keyakinan lebih tinggi.")

        st.write("")
        fig, ax = plt.subplots(figsize=(7, 3))
        classes = list(class_scores.keys())
        values = list(class_scores.values())
        colors = ["#10b981", "#f59e0b", "#ef4444"]
        ax.bar(classes, values, color=colors, edgecolor="#ffffff", linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_title("Distribusi Skor Prediksi per Kelas", fontsize=14, color="#0f172a", pad=14)
        ax.set_ylabel("Skor", color="#475569")
        ax.set_facecolor("#ffffff")
        fig.patch.set_facecolor("#f8fafc")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(colors="#64748b")
        for idx, value in enumerate(values):
            ax.text(idx, value + 0.03, f"{value:.2f}", ha="center", va="bottom", color="#0f172a", fontweight="700")
        st.pyplot(fig)

        st.write("")
        st.markdown(
            """
            <div style='display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px;'>
            """,
            unsafe_allow_html=True,
        )
        dominant = get_dominant_factors(input_data)
        for title, value, description in dominant:
            st.markdown(
                f"""
                <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 22px; padding: 18px;'>
                    <div style='font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>{title}</div>
                    <div style='font-size: 1.4rem; font-weight: 700; color: #2563eb; margin-bottom: 10px;'>{value}</div>
                    <div style='font-size: 0.9rem; color: #475569; line-height: 1.7;'>{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("""
            </div>
            """, unsafe_allow_html=True,
        )

        st.write("")
        if label == "Rendah":
            reco_cards = [
                ("Akademik", "Pertahankan jadwal belajar teratur dan hindari penundaan tugas."),
                ("Gaya Hidup", "Jaga pola tidur yang baik dan tetap konsumsi makanan sehat."),
                ("Sosial", "Terus gunakan dukungan teman atau keluarga untuk menjaga keseimbangan.")
            ]
        elif label == "Sedang":
            reco_cards = [
                ("Akademik", "Buat perencanaan tugas dan bagi beban belajar menjadi bagian yang lebih kecil."),
                ("Gaya Hidup", "Tingkatkan kualitas tidur dan tambahkan waktu istirahat singkat.") ,
                ("Sosial", "Bicarakan tekanan dengan teman, keluarga, atau dosen pembimbing."),
            ]
        else:
            reco_cards = [
                ("Akademik", "Kurangi beban nonprioritas dan minta dukungan akademik jika diperlukan."),
                ("Gaya Hidup", "Fokus pada istirahat cukup, relaksasi, dan olahraga ringan."),
                ("Sosial", "Ceritakan kondisi kepada orang terdekat atau konselor kampus."),
            ]

        st.markdown(
            """
            <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 24px; padding: 24px;'>
                <div style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 16px;'>Rekomendasi Personal</div>
            """,
            unsafe_allow_html=True,
        )
        for title, text in reco_cards:
            st.markdown(
                f"""
                <div style='margin-bottom: 16px;'>
                    <div style='font-size: 0.96rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;'>{title}</div>
                    <div style='font-size: 0.9rem; color: #475569; line-height: 1.7;'>{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("""
            </div>
            """, unsafe_allow_html=True,
        )

        st.write("")
        detail_df = pd.DataFrame({
            "Faktor": [
                "Tingkat Kecemasan",
                "Kualitas Tidur",
                "Beban Belajar",
                "Performa Akademik",
                "Dukungan Sosial",
                "Kekhawatiran Karier",
            ],
            "Nilai": [
                f"{anxiety_level}/20",
                f"{sleep_quality}/5",
                f"{study_load}/5",
                f"{academic_performance}/5",
                f"{social_support}/5",
                f"{future_career_concerns}/5",
            ],
            "Kategori": [
                get_factor_category("anxiety_level", anxiety_level),
                get_factor_category("sleep_quality", sleep_quality),
                get_factor_category("study_load", study_load),
                get_factor_category("academic_performance", academic_performance),
                get_factor_category("social_support", social_support),
                get_factor_category("future_career_concerns", future_career_concerns),
            ],
        })

        st.markdown(
            """
            <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 24px; padding: 22px;'>
                <div style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;'>Detail Input</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
            <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 24px; padding: 34px; box-shadow: 0 20px 50px rgba(15, 23, 42, 0.04);'>
                <div style='font-size: 1.3rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;'>Masukkan data mahasiswa untuk melihat hasil prediksi.</div>
                <p style='color: #475569; font-size: 0.96rem; line-height: 1.75; margin-bottom: 18px;'>Setelah klik prediksi, sistem akan menampilkan label stres, confidence score, faktor paling berpengaruh, dan rekomendasi personal.</p>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;'>
                    <div style='background: #eff6ff; border-radius: 20px; padding: 18px;'>
                        <div style='font-size: 0.95rem; font-weight: 700; color: #2563eb;'>6 Fitur Input</div>
                        <div style='color: #475569; margin-top: 8px;'>Anxiety, tidur, belajar, akademik, sosial, karier.</div>
                    </div>
                    <div style='background: #dbf4ff; border-radius: 20px; padding: 18px;'>
                        <div style='font-size: 0.95rem; font-weight: 700; color: #0f766e;'>3 Kelas Prediksi</div>
                        <div style='color: #475569; margin-top: 8px;'>Rendah, Sedang, Tinggi.</div>
                    </div>
                    <div style='background: #ede9fe; border-radius: 20px; padding: 18px;'>
                        <div style='font-size: 0.95rem; font-weight: 700; color: #6d28d9;'>Output Rekomendasi</div>
                        <div style='color: #475569; margin-top: 8px;'>Saran akademik, gaya hidup, dan dukungan sosial.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.markdown(
    """
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 16px;'>
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 24px; padding: 22px;'>
            <div style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 14px;'>Bagaimana Sistem Bekerja?</div>
            <ul style='color: #475569; line-height: 1.8; margin: 0; padding-left: 18px;'>
                <li>Input data mahasiswa</li>
                <li>Normalisasi data</li>
                <li>Fuzzifikasi adaptif</li>
                <li>Perhitungan firing strength</li>
                <li>Inferensi Neuro-Fuzzy</li>
                <li>Output prediksi dan rekomendasi</li>
            </ul>
            <p style='color: #475569; margin-top: 12px;'>Metode Neuro-Fuzzy menggabungkan kemampuan fuzzy logic yang mudah diinterpretasikan dengan proses pembelajaran dari data.</p>
        </div>
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 24px; padding: 22px;'>
            <div style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 14px;'>Catatan Akademik</div>
            <p style='color: #475569; line-height: 1.8; margin: 0;'>Sistem menggunakan FIS sebagai baseline interpretatif, sedangkan ANFIS menjadi metode utama karena dapat mempelajari pola dari data. Prediksi ini bersifat awal dan tidak menggantikan penilaian profesional.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.markdown(
    """
    <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 24px; padding: 22px;'>
        <div style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 18px;'>Informasi Dataset dan Fitur</div>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;'>
            <div style='background: #eff6ff; border-radius: 20px; padding: 16px;'>
                <div style='font-weight: 700; color: #2563eb;'>Tingkat Kecemasan</div>
                <div style='color: #475569; margin-top: 8px;'>Respons emosional terhadap tekanan studi dan sosial.</div>
            </div>
            <div style='background: #ecfdf5; border-radius: 20px; padding: 16px;'>
                <div style='font-weight: 700; color: #0f766e;'>Kualitas Tidur</div>
                <div style='color: #475569; margin-top: 8px;'>Kualitas istirahat yang memengaruhi energi mental.</div>
            </div>
            <div style='background: #ffedd5; border-radius: 20px; padding: 16px;'>
                <div style='font-weight: 700; color: #ea580c;'>Beban Belajar</div>
                <div style='color: #475569; margin-top: 8px;'>Jumlah tugas dan tekanan akademik yang dirasakan.</div>
            </div>
            <div style='background: #f8fafc; border-radius: 20px; padding: 16px;'>
                <div style='font-weight: 700; color: #0f172a;'>Performa Akademik</div>
                <div style='color: #475569; margin-top: 8px;'>Hasil belajar yang memengaruhi kepercayaan diri.</div>
            </div>
            <div style='background: #ede9fe; border-radius: 20px; padding: 16px;'>
                <div style='font-weight: 700; color: #6d28d9;'>Dukungan Sosial</div>
                <div style='color: #475569; margin-top: 8px;'>Ketersediaan bantuan dari teman, keluarga, atau lingkungan kampus.</div>
            </div>
            <div style='background: #ecfeff; border-radius: 20px; padding: 16px;'>
                <div style='font-weight: 700; color: #0f766e;'>Kekhawatiran Karier</div>
                <div style='color: #475569; margin-top: 8px;'>Kecemasan terhadap masa depan dan prospek pekerjaan.</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
