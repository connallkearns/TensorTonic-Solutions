import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    numerator = np.dot(a, b)
    denominator = np.linalg.norm(a, ord = 2) * np.linalg.norm(b, ord = 2)
    cos_dist = np.divide(numerator, denominator) if denominator != 0 else 0
    return float(cos_dist)