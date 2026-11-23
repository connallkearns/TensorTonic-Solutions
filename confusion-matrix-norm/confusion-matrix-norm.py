import numpy as np

def confusion_matrix_norm(y_true: list, y_pred: list, num_classes: int | None = None, normalize: str = "none") -> np.ndarray:
    """
    Returns the confusion matrix as a NumPy array.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num_classes = np.asarray(range(np.max(y_true) + 1)).size if num_classes is None else num_classes # Default test cases contain null vals, so this finds number of classes
    conf_mat = np.bincount(y_true * num_classes + y_pred) if y_true.size > 0 else np.zeros((num_classes, num_classes))
    conf_mat = np.pad(conf_mat, (0, num_classes ** 2 - conf_mat.size)).reshape(-1, num_classes)
    if normalize == "true":
        return ((conf_mat).T / conf_mat.sum(axis = 1)).T
    elif normalize == "pred":
        return conf_mat / conf_mat.sum(axis = 0)
    elif normalize == "all":
        return conf_mat / conf_mat.sum()
    else:
        return conf_mat