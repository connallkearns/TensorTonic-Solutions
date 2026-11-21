import numpy as np

def manhattan_distance(x: list, y: list) -> float:
    """
    Returns the Manhattan distance as a Python float.
    """
    return np.sum(np.linalg.norm(np.subtract(x, y), ord = 1))