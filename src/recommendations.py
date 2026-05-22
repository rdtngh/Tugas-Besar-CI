def get_recommendation(level: str) -> str:
    """Mengembalikan rekomendasi ringkas berdasarkan tingkat stres."""
    if level == "Rendah":
        return (
            "Kondisi saat ini cukup stabil. Pertahankan rutinitas teratur, pola tidur baik, dan dukungan sosial yang sudah ada."
        )
    if level == "Sedang":
        return (
            "Terdapat tekanan sedang. Perbaiki manajemen waktu, istirahat cukup, dan jaga komunikasi dengan orang terdekat."
        )
    return (
        "Tingkat stres tinggi. Kurangi beban nonprioritas, perbaiki pola istirahat, dan cari dukungan sosial atau konseling kampus."
    )


def get_recommendation_details(level: str) -> dict:
    """Mengembalikan rekomendasi detail berdasarkan level stres."""
    if level == "Rendah":
        return {
            "Akademik": [
                "Pertahankan jadwal belajar teratur.",
                "Bagi tugas besar menjadi bagian kecil agar tetap konsisten.",
            ],
            "Gaya Hidup": [
                "Jaga pola tidur yang stabil.",
                "Lanjutkan aktivitas fisik ringan secara rutin.",
            ],
            "Sosial": [
                "Pertahankan komunikasi dengan teman dan keluarga.",
                "Gunakan dukungan sosial untuk menjaga keseimbangan mental.",
            ],
        }
    if level == "Sedang":
        return {
            "Akademik": [
                "Buat perencanaan tugas mingguan.",
                "Batasi volume belajar dalam satu sesi agar tidak kelelahan.",
            ],
            "Gaya Hidup": [
                "Cukupi waktu tidur dan istirahat pendek setiap hari.",
                "Kurangi konsumsi kafein di malam hari.",
            ],
            "Sosial": [
                "Sampaikan kondisi stres kepada teman dekat atau keluarga.",
                "Pertimbangkan berdiskusi dengan dosen pembimbing jika perlu.",
            ],
        }
    return {
        "Akademik": [
            "Kurangi beban tugas yang tidak mendesak.",
            "Minta bantuan akademik atau konsultasi dengan dosen jika memungkinkan.",
        ],
        "Gaya Hidup": [
            "Atur tidur yang cukup dan waktu istirahat terjadwal.",
            "Lakukan relaksasi ringan seperti peregangan atau berjalan singkat.",
        ],
        "Sosial": [
            "Bercerita kepada orang terdekat atau konselor kampus.",
            "Kurangi aktivitas sosial berlebihan jika terasa menambah tekanan.",
        ],
    }
