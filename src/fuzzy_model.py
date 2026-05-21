"""
Fuzzy Inference System Model untuk prediksi stres mahasiswa.

Modul ini mengintegrasikan membership function dan rule base untuk melakukan
prediksi menggunakan pendekatan Fuzzy Inference System (FIS).
"""

from src.fuzzy_membership import fuzzify_input
from src.fuzzy_rules import evaluate_rules


def defuzzify(class_scores):
    """
    Melakukan defuzzifikasi dengan metode center of gravity (COG).
    Dalam hal ini, cukup mengambil class dengan score tertinggi.
    
    Parameters:
    class_scores: dictionary {'rendah': score, 'sedang': score, 'tinggi': score}
    
    Returns:
    (label_output, confidence_score)
    """
    if all(score == 0 for score in class_scores.values()):
        # Jika semua score 0, gunakan fallback
        return 'Sedang', 0.0
    
    # Ambil class dengan score tertinggi
    label = max(class_scores, key=class_scores.get)
    score = class_scores[label]
    
    # Mapping ke format output
    label_map = {
        'rendah': 'Rendah',
        'sedang': 'Sedang',
        'tinggi': 'Tinggi',
    }
    
    return label_map.get(label, 'Sedang'), score


def fuzzy_predict(input_data):
    """
    Melakukan prediksi menggunakan Fuzzy Inference System.
    
    Parameters:
    input_data: dictionary berisi nilai input
        {
            'anxiety_level': int/float,
            'sleep_quality': int/float,
            'study_load': int/float,
            'academic_performance': int/float,
            'social_support': int/float,
            'future_career_concerns': int/float,
        }
    
    Returns:
    {
        'label': 'Rendah' atau 'Sedang' atau 'Tinggi',
        'score': confidence score (0-1),
        'class_scores': {
            'Rendah': score,
            'Sedang': score,
            'Tinggi': score,
        }
    }
    """
    try:
        # Step 1: Fuzzifikasi input
        fuzzy_input = fuzzify_input(input_data)
        
        # Step 2: Evaluasi rule base
        class_scores = evaluate_rules(fuzzy_input)
        
        # Step 3: Defuzzifikasi
        label, score = defuzzify(class_scores)
        
        # Mapping class scores untuk output
        class_scores_output = {
            'Rendah': class_scores['rendah'],
            'Sedang': class_scores['sedang'],
            'Tinggi': class_scores['tinggi'],
        }
        
        return {
            'label': label,
            'score': score,
            'class_scores': class_scores_output,
        }
    
    except Exception as e:
        # Fallback jika ada error
        print(f"Error dalam fuzzy prediction: {e}")
        return {
            'label': 'Sedang',
            'score': 0.0,
            'class_scores': {'Rendah': 0.0, 'Sedang': 0.5, 'Tinggi': 0.0},
        }


def fuzzy_predict_batch(df):
    """
    Melakukan prediksi pada batch data.
    
    Parameters:
    df: DataFrame berisi fitur input
    
    Returns:
    list of prediction results
    """
    predictions = []
    
    for idx, row in df.iterrows():
        input_data = {
            'anxiety_level': float(row['anxiety_level']),
            'sleep_quality': float(row['sleep_quality']),
            'study_load': float(row['study_load']),
            'academic_performance': float(row['academic_performance']),
            'social_support': float(row['social_support']),
            'future_career_concerns': float(row['future_career_concerns']),
        }
        
        result = fuzzy_predict(input_data)
        predictions.append({
            'label': result['label'],
            'score': result['score'],
            'class_scores': result['class_scores'],
        })
    
    return predictions
