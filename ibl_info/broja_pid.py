import ibl_info.BROJA_2PID as broja
import numpy as np


def build_probability_distribution(Y, X1, X2):
    """
    Generate a trivariate probability distribution for
    one target variable and 2 source variables

    Args:
        Y (np.array): Target variable
        X1 (np.array): Source 1
        X2 (np.array): Source 2
    """

    Y = np.asarray(Y, dtype=np.int16)
    X1 = np.asarray(X1, dtype=np.int16)
    X2 = np.asarray(X2, dtype=np.int16)

    counts = dict()
    n_samples = Y.shape[0]

    for i in range(n_samples):
        if (Y[i], X1[i], X2[i]) in counts.keys():
            counts[(Y[i], X1[i], X2[i])] += 1
        else:
            counts[(Y[i], X1[i], X2[i])] = 1

    pmf = dict()
    for xyz, c in counts.items():
        pmf[xyz] = c / float(n_samples)

    return pmf


def compute_pid(Y, X1, X2):
    """
    Compute the partial information decomposition for one target and 2 sources

    Args:
        Y (np.array): Target variable, decoding variable
        X1 (np.array): Source 1, neuron 1 spike counts
        X2 (np.array): Source 2, neuron 2 spike counts
    """

    dirty_pdf = build_probability_distribution(Y, X1, X2)
    info_decomposition = broja.pid(dirty_pdf, output=0)
    return np.asarray(
        [
            info_decomposition["UIY"],
            info_decomposition["UIZ"],
            info_decomposition["SI"],
            info_decomposition["CI"],
        ]
    )


def split_data(Y, X1, X2, splits):

    permutation_order = np.random.permutation(len(Y))

    # shuffle everything
    Y_shuffled = Y[permutation_order]
    X1_shuffled = X1[permutation_order]
    X2_shuffled = X2[permutation_order]

    # now split into subarrays
    y_partitions = np.array_split(Y_shuffled, splits)
    x1_partitions = np.array_split(X1_shuffled, splits)
    x2_partitions = np.array_split(X2_shuffled, splits)

    return y_partitions, x1_partitions, x2_partitions


def compute_pid_unbiased(Y, X1, X2, repeats=1):
    """
    compute unbiased PID terms using Quadratic estimation
    PID_plugin = PID_true + a/N + b/N2

    Args:
        Y (np.array): target variable
        X1 (np.array): source variable 1
        X2 (np.array): source_variable_2
    """

    n_trials = len(Y)
    n_partitions = np.asarray([1, 2, 4])
    pid_array = np.zeros((len(n_partitions), 4))
    pid_unbiased = np.zeros((4))
    # for complete data, i/e, Iplugin
    pid_array[0, :] = compute_pid(Y, X1, X2)

    # i'll just do this subsampling once, maybe we should do it more times

    for idx in range(1, len(n_partitions)):
        splits = n_partitions[idx]
        # split data splits the data into equal number of splits
        y_part, x1_part, x2_part = split_data(Y, X1, X2, splits)
        for parts in range(0, splits):
            y_temp = y_part[parts]
            x1_temp = x1_part[parts]
            x2_temp = x2_part[parts]
            # now what
            pid_temp = compute_pid(y_temp, x1_temp, x2_temp)

            if splits == 2:
                pid_array[idx, :] = pid_array[idx, :] + pid_temp / 2
            elif splits == 4:
                pid_array[idx, :] = pid_array[idx, :] + pid_temp / 4

    # should have 3 pids completely ready
    # now to run polyfit, etc

    x_extrap = n_partitions / n_trials

    # for each column of pid_array, fit an equation:
    params = np.zeros((4, 3))  # since QE
    for idx in range(0, 4):
        values = pid_array[:, idx]  # 0 is U1, 1 is U2, 2 is SI and 3 is CI
        params[idx, :] = np.polyfit(x_extrap, values, 2)
        pid_unbiased[idx] = params[idx, 2]  # NOTE: divide here when we do more repetations

    return pid_unbiased


def MI(Y, X):
    # TODO: clean this up
    # annoying but quick
    dirty_pdf = build_probability_distribution(Y, X, X)
    MIYX = broja.I_Y(dirty_pdf)
    return MIYX


def unbiasedMI(Y, X, repeats=1):
    n_trials = len(Y)
    n_partitions = np.asarray([1, 2, 4])
    mi_array = np.zeros((len(n_partitions)))
    mi_unbiased = 0
    # for complete data, i/e, Iplugin
    mi_array[0] = MI(Y, X)

    # i'll just do this subsampling once, maybe we should do it more times

    for idx in range(1, len(n_partitions)):
        splits = n_partitions[idx]
        # split data splits the data into equal number of splits
        y_part, x_part, _ = split_data(Y, X, X, splits)
        for parts in range(0, splits):
            y_temp = y_part[parts]
            x_temp = x_part[parts]
            # now what
            mi_temp = MI(y_temp, x_temp)
            if splits == 2:
                mi_array[idx] = mi_array[idx] + mi_temp / 2
            elif splits == 4:
                mi_array[idx] = mi_array[idx] + mi_temp / 4

    # should have 3 pids completely ready
    # now to run polyfit, etc

    x_extrap = n_partitions / n_trials

    # for each column of mi_array, fit an equation:
    params = np.zeros((3))  # since QE

    # 0 is U1, 1 is U2, 2 is SI and 3 is CI
    params = np.polyfit(x_extrap, mi_array, 2)
    mi_unbiased = params[2]  # NOTE: divide here when we do more repetations

    return mi_unbiased


def coinformation(Y, X1, X2):
    """
    Compute co-information and also trivariate mutual information

    Args:
        Y (np.array): Target variable, decoding variable
        X1 (np.array): Source 1, neuron 1 spike counts
        X2 (np.array): Source 2, neuron 2 spike counts

    """

    dirty_pdf = build_probability_distribution(Y, X1, X2)
    MIYX1X2 = broja.I_YZ(dirty_pdf)
    MIYX1 = broja.I_Y(dirty_pdf)
    MIYX2 = broja.I_Z(dirty_pdf)

    return [MIYX1X2 - MIYX1 - MIYX2, MIYX1X2, MIYX1, MIYX2]
