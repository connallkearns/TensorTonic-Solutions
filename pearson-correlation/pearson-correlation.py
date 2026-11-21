import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    X = np.asarray(X)
    N = X.shape[1]
    if N < 2 or len(X.shape) < 2 or X.size == 0:
        return None
    X_centred = X - X.mean(axis = 0)
    covar_mat = (X_centred.T @ X_centred) / (N - 1)
    sigma = np.sqrt(np.diag(covar_mat))
    r = covar_mat / np.outer(sigma, sigma.T)
    return r