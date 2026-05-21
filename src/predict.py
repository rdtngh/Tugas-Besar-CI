"""
Module prediksi untuk sistem prediksi stres mahasiswa.

Menggunakan Fuzzy Inference System (FIS) sebagai metode utama,
dengan fallback ke scoring sederhana jika diperlukan.
"""

from src.fuzzy_model import fuzzy_predict


def predict_stress(input_data: dict):
    """
    Melakukan prediksi tingkat stres menggunakan Fuzzy Inference System.
    
    Parameters:
    input_data: dictionary berisi nilai input
        {
            'anxiety_level': int/float (0-20),
            'sleep_quality': int/float (0-5),
            'study_load': int/float (0-5),
            'academic_performance': int/float (0-5),
            'social_support': int/float (0-5),
            'future_career_concerns': int/float (0-5),
        }
    
    Returns:
    (label, score, class_scores)
    - label: 'Rendah', 'Sedang', atau 'Tinggi'
    - score: confidence score (0-1)
    - class_scores: dict berisi skor untuk setiap kelas
    """
    try:
        # Gunakan Fuzzy Inference System
        result = fuzzy_predict(input_data)
        
        label = result['label']
        score = result['score']
        class_scores = result['class_scores']
        
        return label, score, class_scores
    
    except Exception as e:
        # Fallback jika ada error
        print(f"Error dalam fuzzy prediction: {e}")
        return 'Sedang', 0.0, {
            'Rendah': 0.0,
            'Sedang': 0.5,
            'Tinggi': 0.0,
        }
