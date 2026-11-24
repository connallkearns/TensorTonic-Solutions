import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    train_val_idx: list = list()
    if shuffle: # Checks whether to shuffle indices
        indices: np.array = np.random.default_rng(seed).permutation(np.arange(N)) # Sets user-supplied seed and shuffles
    else: 
        indices: np.array = np.arange(N) # Generates indices
    for fold in range(k):
        indices_split = np.array_split(indices, k) # Splits array by `k`
        val_idx = indices_split.pop(fold) # `pop` extract validation set
        train_idx = np.concat(indices_split) # Concatenates into a single array
        train_val_idx.append({"train_idx": train_idx,
                             "val_idx": val_idx})
    return train_val_idx