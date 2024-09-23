
import numpy as np
from broja_pid import compute_pid, coinformation


def permutation_test_coinformation(neural_data, target_variable, repeats=50):
    """
        Cluster-permutation test for shared, redundant information

    Args:
        neural_data (np.array): neurons x trials
        target_variable (np.array): decoding variable/ variable of interest
        repeats (int, optional): number of times to run the computation. Defaults to 50.
    """

    for r in repeats:
        yprime = np.random.permutation(target_variable)

    return NotImplementedError