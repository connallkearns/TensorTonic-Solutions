def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    precision = len([rel for rel in relevant if rel in set(recommended[:k])]) / k
    recall = len([rel for rel in relevant if rel in set(recommended[:k])]) / len(relevant)
    return [precision, recall]