import streamlit as st
from src.predict import predict_stress
from src.recommendations import get_recommendation

st.set_page_config(
    page_title="Sistem Prediksi Tingkat Stres Mahasiswa",
    page_icon="??",
    layout="centered",
)

st.title("Sistem Prediksi Tingkat Stres Mahasiswa")
st.markdown(
    "Aplikasi ini membantu memprediksi level stres mahasiswa berdasarkan beberapa indikator gaya hidup dan akademik. "
    "Hasil prediksi bersifat informasi awal dan bukan diagnosis medis."
)

st.markdown("---")

with st.sidebar:
    st.header("Masukkan data")
    anxiety_level = st.slider("Tingkat Kecemasan", 0, 20, 10)
    sleep_quality = st.slider("Kualitas Tidur", 0, 5, 3)
    study_load = st.slider("Beban Belajar", 0, 5, 3)
    academic_performance = st.slider("Performa Akademik", 0, 5, 3)
    social_support = st.slider("Dukungan Sosial", 0, 5, 3)
    future_career_concerns = st.slider("Kekhawatiran Karier", 0, 5, 3)
    st.markdown("---")
    st.write("**Catatan:** Sistem ini hanya alat bantu prediksi awal dan bukan diagnosis medis.")

input_data = {
    "anxiety_level": anxiety_level,
    "sleep_quality": sleep_quality,
    "study_load": study_load,
    "academic_performance": academic_performance,
    "social_support": social_support,
    "future_career_concerns": future_career_concerns,
}

if st.button("Prediksi Tingkat Stres"):
    label, score = predict_stress(input_data)
    recommendation = get_recommendation(label)

    if label == "Rendah":
        st.success(f"Prediksi: {label}")
    elif label == "Sedang":
        st.warning(f"Prediksi: {label}")
    else:
        st.error(f"Prediksi: {label}")

    if score is not None:
        st.write(f"Skor prediksi: {score:.2f}")

    st.subheader("Rekomendasi")
    st.write(recommendation)

    with st.expander("Detail input"):
        st.json(input_data)

st.markdown("---")
st.write("**Cara kerja sistem:** Aplikasi memproses input studi dan gaya hidup, lalu memprediksi tingkat stres sebagai Rendah, Sedang, atau Tinggi.")
