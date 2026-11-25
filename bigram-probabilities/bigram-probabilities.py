import numpy as np

def bigram_probabilities(tokens: list) -> dict:
    """
    Returns a dictionary with vocab, counts, and probabilities.
    """
    vocab = np.unique(tokens) # Finds unique tokens
    unigrams = {k: 0 for k in vocab} # Defines unigram dict
    for token in tokens: # Inits for-loop for tokens
        unigrams[token] += 1 # Find unigrams
    bigrams_mat = np.zeros((vocab.size, vocab.size)) # Initialises bigrams mat
    for token_1, token_2 in zip(tokens[:-1], tokens[1:]):
        token_1_idx, token_2_idx = np.where(vocab == token_1), np.where(vocab == token_2) # Gets token indexes
        bigrams_mat[token_1_idx, token_2_idx] += 1 # Adds to matrix
    bigram_probs = (bigrams_mat + 1) / (bigrams_mat.sum(axis = 1, keepdims=True) + vocab.size) # Calcs bigram probs
    return {"vocab": vocab.tolist(), 
           "counts": bigrams_mat,
           "probabilities": bigram_probs}