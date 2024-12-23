import numpy as np


def fair_sharer(values: list, num_iterations: int = 1, share: float = 0.1):
    """Runs num_iterations.
    In each iteration the highest value in values gives a fraction (share)
    to both the left and right neighbor. The leftmost field is considered
    the neightbor of the rightmost field.
    Examples:
    fair_sharer([0, 1000, 800, 0], 1) --> [100, 800, 900, 0]
    fair_sharer([0, 1000, 800, 0], 2) --> [100, 890, 720, 90]
    Args
    values:
    1D array of values (list or numpy array)
    num_iteration:
    Integer to set the number of iterations
    """

    #Important: np.matrix + np.ndarray will be flattened to 1D list
    if not isinstance(values, (list, np.ndarray, np.matrix)):
        raise TypeError(f"Object of unsupported type {type(values)}.")
    if not isinstance(num_iterations, (int, np.integer)):
        raise TypeError(f"Object of unsupported type {type(num_iterations)}.")
    if not isinstance(share, float) or not np.issubdtype(type(share), np.floating):
        raise TypeError(f"Object of unsupported type {type(share)}.")
    values = list_converter(values)

    #actual function
    for _ in range(num_iterations):
        max_value = max(values)
        i_max_val = values.index(max_value)
        share_real = max_value * share
        values[i_max_val] -= 2*share_real
        values[(i_max_val + 1) % len(values) ] += share_real
        values[(i_max_val - 1) % len(values) ] += share_real
    values_new = values
    return list(values_new)

def list_converter(object):
    """Flattens numpy arrays and matrices, converts them to lists."""
    if isinstance(object, (np.ndarray, np.matrix)):
        object = np.asarray(object) #converts matrix to array
        object = object.flatten()
        object = object.tolist()
        print("Warning: Numpy array or matrix has been converted to list. Check dimensionality of input-object: might have been unintentionally flattened.")
    return object

if __name__ == "__main__":
    array = np.matrix([[0, 0], [1000, 5], [800, 0], [0,0]])
    print(fair_sharer(values=array, num_iterations=2))