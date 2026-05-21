"""
Fuzzy Rule Base untuk sistem prediksi stres mahasiswa.

Modul ini mendefinisikan rule-rule IF-THEN dalam bentuk yang mudah diproses.
Rule base terdiri dari 12+ aturan yang menggunakan operator AND (min).
"""


# Definisi rule base dalam format Python
FUZZY_RULES = [
    # ========== Aturan untuk Stress Level TINGGI ==========
    {
        'name': 'R1: Anxiety tinggi & Sleep buruk => Stress tinggi',
        'conditions': [
            ('anxiety_level', 'tinggi'),
            ('sleep_quality', 'buruk'),
        ],
        'operator': 'AND',  # min
        'output': 'tinggi',
    },
    {
        'name': 'R2: Anxiety tinggi & Study berat => Stress tinggi',
        'conditions': [
            ('anxiety_level', 'tinggi'),
            ('study_load', 'berat'),
        ],
        'operator': 'AND',
        'output': 'tinggi',
    },
    {
        'name': 'R3: Anxiety tinggi & Career concerns tinggi => Stress tinggi',
        'conditions': [
            ('anxiety_level', 'tinggi'),
            ('future_career_concerns', 'tinggi'),
        ],
        'operator': 'AND',
        'output': 'tinggi',
    },
    {
        'name': 'R4: Study berat & Academic rendah => Stress tinggi',
        'conditions': [
            ('study_load', 'berat'),
            ('academic_performance', 'rendah'),
        ],
        'operator': 'AND',
        'output': 'tinggi',
    },
    {
        'name': 'R5: Social support rendah & Anxiety tinggi => Stress tinggi',
        'conditions': [
            ('social_support', 'rendah'),
            ('anxiety_level', 'tinggi'),
        ],
        'operator': 'AND',
        'output': 'tinggi',
    },
    {
        'name': 'R6: Sleep buruk & Career concerns tinggi => Stress tinggi',
        'conditions': [
            ('sleep_quality', 'buruk'),
            ('future_career_concerns', 'tinggi'),
        ],
        'operator': 'AND',
        'output': 'tinggi',
    },
    
    # ========== Aturan untuk Stress Level SEDANG ==========
    {
        'name': 'R7: Anxiety sedang & Study sedang => Stress sedang',
        'conditions': [
            ('anxiety_level', 'sedang'),
            ('study_load', 'sedang'),
        ],
        'operator': 'AND',
        'output': 'sedang',
    },
    {
        'name': 'R8: Sleep sedang & Academic sedang => Stress sedang',
        'conditions': [
            ('sleep_quality', 'sedang'),
            ('academic_performance', 'sedang'),
        ],
        'operator': 'AND',
        'output': 'sedang',
    },
    {
        'name': 'R9: Career concerns sedang & Social support sedang => Stress sedang',
        'conditions': [
            ('future_career_concerns', 'sedang'),
            ('social_support', 'sedang'),
        ],
        'operator': 'AND',
        'output': 'sedang',
    },
    {
        'name': 'R10: Study sedang & Sleep sedang => Stress sedang',
        'conditions': [
            ('study_load', 'sedang'),
            ('sleep_quality', 'sedang'),
        ],
        'operator': 'AND',
        'output': 'sedang',
    },
    
    # ========== Aturan untuk Stress Level RENDAH ==========
    {
        'name': 'R11: Anxiety rendah & Sleep baik => Stress rendah',
        'conditions': [
            ('anxiety_level', 'rendah'),
            ('sleep_quality', 'baik'),
        ],
        'operator': 'AND',
        'output': 'rendah',
    },
    {
        'name': 'R12: Social support tinggi & Academic tinggi => Stress rendah',
        'conditions': [
            ('social_support', 'tinggi'),
            ('academic_performance', 'tinggi'),
        ],
        'operator': 'AND',
        'output': 'rendah',
    },
    {
        'name': 'R13: Study ringan & Career concerns rendah => Stress rendah',
        'conditions': [
            ('study_load', 'ringan'),
            ('future_career_concerns', 'rendah'),
        ],
        'operator': 'AND',
        'output': 'rendah',
    },
    {
        'name': 'R14: Anxiety rendah & Social support tinggi => Stress rendah',
        'conditions': [
            ('anxiety_level', 'rendah'),
            ('social_support', 'tinggi'),
        ],
        'operator': 'AND',
        'output': 'rendah',
    },
]


def evaluate_rules(fuzzy_input):
    """
    Mengevaluasi seluruh rule base terhadap fuzzy input.
    
    Parameters:
    fuzzy_input: dictionary dari fuzzify_input() berisi derajat keanggotaan
    
    Returns:
    dictionary berisi firing strength untuk setiap output class:
    {
        'rendah': max_firing_strength,
        'sedang': max_firing_strength,
        'tinggi': max_firing_strength,
    }
    """
    class_scores = {
        'rendah': 0.0,
        'sedang': 0.0,
        'tinggi': 0.0,
    }
    
    # Evaluasi setiap rule
    for rule in FUZZY_RULES:
        # Hitung antecedent (premise) menggunakan operator AND = min
        firing_strength = 1.0
        
        for feature_name, fuzzy_set in rule['conditions']:
            # Ambil derajat keanggotaan fitur pada fuzzy set tertentu
            membership_degree = fuzzy_input[feature_name][fuzzy_set]
            
            if rule['operator'] == 'AND':
                # AND = minimum
                firing_strength = min(firing_strength, membership_degree)
        
        # Tentukan output dan update score untuk class tersebut
        output_class = rule['output']
        class_scores[output_class] = max(class_scores[output_class], firing_strength)
    
    return class_scores


def print_rules():
    """Menampilkan daftar rule base untuk dokumentasi."""
    print("=" * 70)
    print("FUZZY RULE BASE - Sistem Prediksi Tingkat Stres Mahasiswa")
    print("=" * 70)
    for i, rule in enumerate(FUZZY_RULES, 1):
        print(f"\n{rule['name']}")
        print(f"  Kondisi:")
        for feat, fset in rule['conditions']:
            print(f"    - {feat} = {fset}")
        print(f"  Output: stress_level = {rule['output']}")
    print("\n" + "=" * 70)
