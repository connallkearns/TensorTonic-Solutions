import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    A_np = np.asarray(A)
    A_t_shape = (A_np.shape[1], A_np.shape[0]) 
    A_t = np.zeros(A_t_shape)
    for i, row in enumerate(A):
        for j, item in enumerate(row):
            A_t[j][i] = item
    return A_t