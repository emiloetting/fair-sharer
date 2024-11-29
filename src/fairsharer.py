import numpy as np
import ruff as rf
import pytest 


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
    if type(values) not in [list, np.ndarray, np.linalg.matrix] or type(num_iterations) != int or type(share) != float:
        raise TypeError("Fehlerbeschreibung")
    for _ in range(num_iterations):
        max_value = max(values)
        i_max_val = values.index(max_value)
        share_real = max_value * share
        values[i_max_val] -= 2*share_real
        values[(i_max_val + 1) % len(values) ] += share_real
        values[(i_max_val - 1) % len(values) ] += share_real
    values_new = values
    return values_new

if __name__ == "__main__":
    print(fair_sharer([0, 1000, 800, 0], num_iterations=2))