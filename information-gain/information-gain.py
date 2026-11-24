import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    split_mask = np.asarray(split_mask, dtype = bool) # Convert to numpy array
    split_left = np.asarray(y)[~split_mask] # Gets left split
    split_mask_left = np.where(~split_mask) # Gets left split indices
    split_right = np.asarray(y)[split_mask] # Gets right split
    split_mask_right = np.where(split_mask) # Gets right split indices
    h_y = [[ys for ys in y if ys == curr_class] for curr_class in set(y)]
    h_y = [len(ys) / len(y) for ys in h_y]
    h_y = -sum([p_c * np.log2(p_c) for p_c in h_y])
    h_y_l = [[y_l for y_l in split_left if y_l == label] for label in np.unique(split_left)]
    h_y_l = [len(y_l_subset) / len(split_left) for y_l_subset in h_y_l]
    h_y_l = -sum([p_c * np.log2(p_c) for p_c in h_y_l])
    h_y_r = [[y_r for y_r in split_right if y_r == label] for label in np.unique(split_right)]
    h_y_r = [len(y_r_subset) / len(split_right) for y_r_subset in h_y_r]
    h_y_r = -sum([p_c * np.log2(p_c) for p_c in h_y_r])
    ig = h_y - (len(split_left) / len(y) * h_y_l) - (len(split_right) / len(y) * h_y_r)
    return ig