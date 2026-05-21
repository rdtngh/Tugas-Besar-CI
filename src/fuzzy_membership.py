"""
Fuzzy Membership Functions untuk sistem prediksi stres.

Modul ini menyediakan fungsi untuk menghitung derajat keanggotaan (membership degree)
dari nilai input terhadap himpunan fuzzy yang telah didefinisikan.
"""


def triangular_membership(x, a, b, c):
    """
    Menghitung derajat keanggotaan triangular (segitiga).
    
    Parameters:
    x: nilai input
    a, b, c: parameter triangular dengan a < b < c
    b adalah puncak (membership = 1)
    
    Returns:
    derajat keanggotaan [0, 1]
    """
    if x < a or x > c:
        return 0.0
    elif x <= b:
        if a == b:
            return 1.0
        return (x - a) / (b - a)
    else:
        if b == c:
            return 1.0
        return (c - x) / (c - b)


def trapezoidal_membership(x, a, b, c, d):
    """
    Menghitung derajat keanggotaan trapezoidal (trapesium).
    
    Parameters:
    x: nilai input
    a, b, c, d: parameter trapezoidal dengan a <= b <= c <= d
    [b, c] adalah area dengan membership = 1
    
    Returns:
    derajat keanggotaan [0, 1]
    """
    if x < a or x > d:
        return 0.0
    elif x <= b:
        if a == b:
            return 1.0
        return (x - a) / (b - a)
    elif x <= c:
        return 1.0
    else:
        if c == d:
            return 1.0
        return (d - x) / (d - c)


def fuzzify_input(input_data):
    """
    Mengubah nilai input crisp menjadi derajat keanggotaan fuzzy.
    
    Parameters:
    input_data: dictionary berisi nilai input
        {
            'anxiety_level': int/float (0-20),
            'sleep_quality': int/float (0-5),
            'study_load': int/float (0-5),
            'academic_performance': int/float (0-5),
            'social_support': int/float (0-5),
            'future_career_concerns': int/float (0-5)
        }
    
    Returns:
    dictionary berisi derajat keanggotaan untuk setiap fitur
    {
        'anxiety_level': {'rendah': ..., 'sedang': ..., 'tinggi': ...},
        'sleep_quality': {'buruk': ..., 'sedang': ..., 'baik': ...},
        ...
    }
    """
    fuzzy_input = {}
    
    # Anxiety Level (range 0-20)
    anxiety = input_data['anxiety_level']
    fuzzy_input['anxiety_level'] = {
        'rendah': trapezoidal_membership(anxiety, 0, 0, 5, 9),
        'sedang': triangular_membership(anxiety, 6, 10, 14),
        'tinggi': trapezoidal_membership(anxiety, 11, 15, 20, 20),
    }
    
    # Sleep Quality (range 0-5)
    sleep = input_data['sleep_quality']
    fuzzy_input['sleep_quality'] = {
        'buruk': trapezoidal_membership(sleep, 0, 0, 1.5, 2.5),
        'sedang': triangular_membership(sleep, 1.5, 3, 4),
        'baik': trapezoidal_membership(sleep, 3.2, 4, 5, 5),
    }
    
    # Study Load (range 0-5)
    study = input_data['study_load']
    fuzzy_input['study_load'] = {
        'ringan': trapezoidal_membership(study, 0, 0, 1.5, 2.5),
        'sedang': triangular_membership(study, 1.5, 3, 4),
        'berat': trapezoidal_membership(study, 3.2, 4, 5, 5),
    }
    
    # Academic Performance (range 0-5)
    perf = input_data['academic_performance']
    fuzzy_input['academic_performance'] = {
        'rendah': trapezoidal_membership(perf, 0, 0, 1.5, 2.5),
        'sedang': triangular_membership(perf, 1.5, 3, 4),
        'tinggi': trapezoidal_membership(perf, 3.2, 4, 5, 5),
    }
    
    # Social Support (range 0-5)
    social = input_data['social_support']
    fuzzy_input['social_support'] = {
        'rendah': trapezoidal_membership(social, 0, 0, 1.5, 2.5),
        'sedang': triangular_membership(social, 1.5, 3, 4),
        'tinggi': trapezoidal_membership(social, 3.2, 4, 5, 5),
    }
    
    # Future Career Concerns (range 0-5)
    career = input_data['future_career_concerns']
    fuzzy_input['future_career_concerns'] = {
        'rendah': trapezoidal_membership(career, 0, 0, 1.5, 2.5),
        'sedang': triangular_membership(career, 1.5, 3, 4),
        'tinggi': trapezoidal_membership(career, 3.2, 4, 5, 5),
    }
    
    return fuzzy_input
