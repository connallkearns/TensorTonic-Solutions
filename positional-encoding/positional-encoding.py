import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    encodings = np.arange(seq_len * d_model, dtype=np.float32).reshape(seq_len, d_model)
    even_cols = np.where(encodings[:, 0::2]) # Get even indexes
    even_pos, even_i = even_cols # Split indexes for easier readability
    odd_cols = np.where(encodings[:, 1::2]) # Get odd indexes
    odd_pos, odd_i = odd_cols
    encodings[:, 0::2][even_cols] = np.sin(even_pos / (base ** ((2 * even_i) / d_model))) # Find even encodings
    encodings[:, 1::2][odd_cols] = np.cos(odd_pos / (base ** ((2 * odd_i) / d_model))) # Find odd encodings
    return encodings # Return the encodings