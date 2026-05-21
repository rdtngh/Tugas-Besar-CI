
def get_recommendation(level: str) -> str:
    """Mengembalikan rekomendasi berdasarkan tingkat stres."""
    if level == "Rendah":
        return (
            "Pertahankan pola tidur yang baik, kelola waktu dengan seimbang, dan lakukan aktivitas positif seperti olahraga ringan atau hobi."
        )
    if level == "Sedang":
        return (
            "Perbaiki jadwal belajar, tingkatkan kualitas tidur, dan kurangi beban aktivitas yang berlebihan. "
            "Luangkan waktu istirahat dan berbicara dengan teman atau keluarga."
        )
    return (
        "Istirahat cukup, bercerita kepada orang terdekat, dan pertimbangkan layanan konseling kampus jika diperlukan. "
        "Fokuskan pada pengurangan tekanan dan pemulihan diri."
    )
