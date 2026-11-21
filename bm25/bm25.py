import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    Returns a NumPy array with one score per document.
    """
    query_bow = set(query_tokens) # Bag of words query
    doc_bow = Counter()
    N: int = len(docs) # Gets number of documents
    doc_count = np.sum([len(doc) for doc in docs])
    doc_len = [len(doc) for doc in docs]
    avgdl = np.mean([len(doc) for doc in docs])
    df = {k: 0 for k in query_bow}
    tf = {k: {i: 0 for i in range(N)} for k in query_bow}
    for i, doc in enumerate(docs): # Loops through docs to get BOW rep
        doc_bow += Counter(doc)
        for token in query_tokens:
            df[token] += 1 if token in doc else 0
            tf[token][i] += Counter(docs[i])[token]
    idf = {k: 0 for k in query_bow}
    for term in query_bow:
        idf[term] = np.log(np.divide(N - df[term] + 0.5, df[term] + 0.5) + 1)
    score = np.zeros(N, dtype = np.float32)
    for doc in range(N):
        for token in query_tokens:
            score[doc] += idf[token] * np.divide(tf[token][doc] * (k1 + 1), tf[token][doc] + k1 * (1 - b + b * np.divide(doc_len[doc], avgdl)))
    return score