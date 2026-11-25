def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    common = 0
    for rel in relevant:
        if rel in set(recommended[:k]):
            common += 1
    precision = common / k
    recall = common / len(relevant)
    return [precision, recall]