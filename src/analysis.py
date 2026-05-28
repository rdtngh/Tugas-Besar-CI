from typing import Dict, List


def classify_input_level(feature_name: str, value: float) -> str:
    """Convert raw input values into interpretive categories."""
    if feature_name == "anxiety_level":
        if value <= 6:
            return "rendah"
        if value <= 13:
            return "sedang"
        return "tinggi"

    if feature_name == "sleep_quality":
        if value <= 2:
            return "buruk"
        if value == 3:
            return "cukup"
        return "baik"

    if feature_name == "study_load":
        if value <= 2:
            return "ringan"
        if value == 3:
            return "sedang"
        return "berat"

    if feature_name == "academic_performance":
        if value <= 2:
            return "rendah"
        if value == 3:
            return "sedang"
        return "baik"

    if feature_name == "social_support":
        if value <= 2:
            return "rendah"
        if value == 3:
            return "sedang"
        return "tinggi"

    if feature_name == "future_career_concerns":
        if value <= 2:
            return "rendah"
        if value == 3:
            return "sedang"
        return "tinggi"

    return "sedang"


def get_factor_contribution(input_data: Dict[str, float]) -> List[Dict[str, str]]:
    """Return factors that most contribute to stress based on user input."""
    factors = [
        {
            "key": "anxiety_level",
            "factor": "Kecemasan",
            "value": input_data["anxiety_level"],
            "level": classify_input_level("anxiety_level", input_data["anxiety_level"]),
            "reason": "Tingkat kecemasan yang tinggi dapat memperbesar beban emosional dan menambah tekanan mental.",
        },
        {
            "key": "sleep_quality",
            "factor": "Kualitas Tidur",
            "value": input_data["sleep_quality"],
            "level": classify_input_level("sleep_quality", input_data["sleep_quality"]),
            "reason": "Kualitas tidur yang buruk bisa menghambat pemulihan fisik dan membuat stres menjadi lebih sulit dikendalikan.",
        },
        {
            "key": "study_load",
            "factor": "Beban Belajar",
            "value": input_data["study_load"],
            "level": classify_input_level("study_load", input_data["study_load"]),
            "reason": "Beban belajar yang berat meningkatkan tekanan akademik dan jangka waktu pengelolaan tugas.",
        },
        {
            "key": "academic_performance",
            "factor": "Performa Akademik",
            "value": input_data["academic_performance"],
            "level": classify_input_level("academic_performance", input_data["academic_performance"]),
            "reason": "Performa akademik yang rendah dapat memperkuat kekhawatiran tentang hasil studi dan kontribusi terhadap stres.",
        },
        {
            "key": "social_support",
            "factor": "Dukungan Sosial",
            "value": input_data["social_support"],
            "level": classify_input_level("social_support", input_data["social_support"]),
            "reason": "Dukungan sosial yang rendah membuat Anda lebih rentan terhadap tekanan karena kurangnya lingkungan pendukung.",
        },
        {
            "key": "future_career_concerns",
            "factor": "Kekhawatiran Karier",
            "value": input_data["future_career_concerns"],
            "level": classify_input_level("future_career_concerns", input_data["future_career_concerns"]),
            "reason": "Kekhawatiran terhadap masa depan dapat menambah beban psikologis dan perasaan ketidakpastian.",
        },
    ]

    def severity(item: Dict[str, str]) -> int:
        if item["key"] == "anxiety_level":
            return 3 if item["level"] == "tinggi" else 2 if item["level"] == "sedang" else 1
        if item["key"] == "sleep_quality":
            return 3 if item["level"] == "buruk" else 2 if item["level"] == "cukup" else 1
        if item["key"] == "study_load":
            return 3 if item["level"] == "berat" else 2 if item["level"] == "sedang" else 1
        if item["key"] == "academic_performance":
            return 3 if item["level"] == "rendah" else 2 if item["level"] == "sedang" else 1
        if item["key"] == "social_support":
            return 3 if item["level"] == "rendah" else 2 if item["level"] == "sedang" else 1
        if item["key"] == "future_career_concerns":
            return 3 if item["level"] == "tinggi" else 2 if item["level"] == "sedang" else 1
        return 1

    ranked = sorted(factors, key=lambda item: (severity(item), item["value"]), reverse=True)
    return [
        {
            "factor": item["factor"],
            "value": item["value"],
            "level": item["level"],
            "reason": item["reason"],
        }
        for item in ranked[:3]
    ]


def generate_analysis_text(input_data: Dict[str, float], prediction_label: str, class_scores: Dict[str, float] = None) -> str:
    """Create the main analysis paragraph based on input and prediction result."""
    contributions = get_factor_contribution(input_data)
    if contributions:
        factor_names = [item["factor"].lower() for item in contributions]
        if len(factor_names) == 1:
            factors_description = factor_names[0]
        else:
            factors_description = ", ".join(factor_names[:-1]) + " dan " + factor_names[-1]

        category_phrases = [
            f"{item['factor'].lower()} berada pada tingkat {item['level']}" for item in contributions
        ]
        category_sentence = ". ".join(category_phrases) + "."
    else:
        factors_description = "indikator input"
        category_sentence = "Nilai input menunjukkan kondisi yang mempengaruhi prediksi stres." 

    if prediction_label == "Tinggi":
        return (
            f"Hasil prediksi menunjukkan tingkat stres tinggi. Hal ini dipengaruhi oleh {factors_description}. "
            f"{category_sentence} "
            "Kombinasi faktor psikologis, akademik, dan gaya hidup tersebut menunjukkan adanya tekanan yang cukup besar terhadap kondisi mahasiswa."
        )

    if prediction_label == "Sedang":
        return (
            f"Hasil prediksi menunjukkan tingkat stres sedang. Beberapa faktor seperti {factors_description} mulai menunjukkan pengaruh terhadap tingkat stres. "
            f"{category_sentence} "
            "Kondisi ini perlu diperhatikan agar tidak berkembang menjadi stres yang lebih tinggi."
        )

    return (
        f"Hasil prediksi menunjukkan tingkat stres rendah. Input yang diberikan menunjukkan bahwa beberapa faktor seperti {factors_description} masih berada pada tingkat yang terkendali. "
        f"{category_sentence} "
        "Pertahankan kebiasaan positif dan terus amati beban akademik, kualitas tidur, serta dukungan sosial Anda."
    )


def generate_dynamic_recommendations(input_data: Dict[str, float], prediction_label: str) -> List[str]:
    """Generate personalized recommendations based on input conditions."""
    recommendations = []
    seen = set()

    def add_recommendation(text: str) -> None:
        if text not in seen and len(recommendations) < 4:
            seen.add(text)
            recommendations.append(text)

    if classify_input_level("anxiety_level", input_data["anxiety_level"]) == "tinggi":
        add_recommendation(
            "Lakukan teknik relaksasi sederhana atau diskusikan kondisi dengan orang terpercaya."
        )

    if classify_input_level("sleep_quality", input_data["sleep_quality"]) == "buruk":
        add_recommendation(
            "Perbaiki pola tidur dan usahakan waktu istirahat yang konsisten."
        )

    if classify_input_level("study_load", input_data["study_load"]) == "berat":
        add_recommendation(
            "Buat prioritas tugas dan bagi pekerjaan menjadi bagian kecil."
        )

    if classify_input_level("academic_performance", input_data["academic_performance"]) == "rendah":
        add_recommendation(
            "Evaluasi strategi belajar dan minta bantuan akademik jika diperlukan."
        )

    if classify_input_level("social_support", input_data["social_support"]) == "rendah":
        add_recommendation(
            "Coba bangun komunikasi dengan teman, keluarga, atau lingkungan kampus."
        )

    if classify_input_level("future_career_concerns", input_data["future_career_concerns"]) == "tinggi":
        add_recommendation(
            "Buat rencana karier bertahap agar kekhawatiran masa depan lebih terarah."
        )

    if len(recommendations) < 4:
        if prediction_label == "Sedang":
            add_recommendation(
                "Atur jeda istirahat secara teratur dan hindari belajar terlalu larut malam."
            )
        elif prediction_label == "Tinggi":
            add_recommendation(
                "Pertimbangkan untuk berbagi beban dengan teman, dosen, atau konselor kampus sehingga stres tidak menumpuk."
            )
        elif prediction_label == "Rendah":
            add_recommendation(
                "Pertahankan pola yang baik dengan tidur cukup dan tetap kelola beban akademik secara bertahap."
            )

    return recommendations[:4]
