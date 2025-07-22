### write code for the following measures
### 1. MI
### 2. PID with broja
### 3. MI Corrections : Linear, Quadratic Estimations
### 4. PID Corrections : Linear, Quadratic Estimations

import numpy as np
from collections import Counter
import ibl_info.measures.BROJA_2PID as broja


def mi_plugin(source, target):
    """
    Computes mutual information between source and target

    Args:
        source (np.array): values for source
        target (np.array): values for target
    """

    pmf = compute_probability_distribution(source, target)
    # now we compute the mi

    # mutual information I(X,Y) = sum( p(x,y)*log(p(x,y)/(p(x)*p(y))) )

    marginal_x = compute_probability_distribution(source)
    marginal_y = compute_probability_distribution(target)

    mi = 0
    for xy, prob in pmf.items():
        x, y = xy
        mi += prob * np.log2(prob / (marginal_x[x] * marginal_y[y]))

    return mi


def entropy(distribution):

    entropy = 0
    for value, prob in distribution.items():
        if prob > 0:
            entropy += prob * np.log2(prob)

    return -entropy


def compute_probability_distribution(variable_one, variable_two=None, variable_three=None):

    if variable_two is None:
        variables = variable_one
    elif variable_three is None:
        variables = list(zip(variable_one, variable_two))
    else:
        variables = list(zip(variable_one, variable_two, variable_three))
    counts = Counter(variables)

    total_observations = variable_one.shape[0]
    pmf = {value: count / total_observations for value, count in counts.items()}

    return pmf


def transfer_entropy(source, target, lag):
    pass


def pid_plugin(source_a, source_b, target):

    pdf = compute_probability_distribution(target, source_a, source_b)

    information_decomposition = broja.pid(pdf, output=0)

    return np.asarray(
        [
            information_decomposition["UIY"],
            information_decomposition["UIZ"],
            information_decomposition["SI"],
            information_decomposition["CI"],
        ]
    )


def FIT(source_a, source_b, target):
    pass


def conditional_transfer_entropy(source_a, source_b, target):
    pass


def split_data(target, sourcea, sourceb=None, splits=0):

    permutation_order = np.random.permutation(len(target))

    # this is not shuffling, rather permuting
    # this ensures essentially not the same part gets picked up in different iterations
    target_shuffled = target[permutation_order]
    sourcea_shuffled = sourcea[permutation_order]
    if sourceb is not None:
        sourceb_shuffled = sourceb[permutation_order]

    # now split into subarrays
    y_partitions = np.array_split(target_shuffled, splits)
    xa_partitions = np.array_split(sourcea_shuffled, splits)
    if sourceb is not None:
        xb_partitions = np.array_split(sourceb_shuffled, splits)

    if sourceb is None:
        return y_partitions, xa_partitions
    else:
        return y_partitions, xa_partitions, xb_partitions


def mi_unbiased(source, target, fit="quadratic", repeats=5):

    n_trials = len(target)
    n_partitions = np.asarray([1, 2, 4])
    mi_array = np.zeros((len(n_partitions)))
    mi_unbiased = 0
    # for complete data, i/e, Iplugin
    mi_array[0] = mi_plugin(source, target)

    for rxs in range(0, repeats):
        # just one repetation if repeats = 1

        for idx in range(1, len(n_partitions)):
            splits = n_partitions[idx]
            # split data splits the data into equal number of splits
            y_part, x_part = split_data(target, source, splits=splits)
            for parts in range(0, splits):
                y_temp = y_part[parts]
                x_temp = x_part[parts]
                # now what
                mi_temp = mi_plugin(x_temp, y_temp)
                if splits == 2:
                    mi_array[idx] = mi_array[idx] + mi_temp / 2
                elif splits == 4:
                    mi_array[idx] = mi_array[idx] + mi_temp / 4

    # i think the division by repeats should happen here
    # not the first one because we do it only once
    mi_array[1] = mi_array[1] / repeats
    mi_array[2] = mi_array[2] / repeats

    x_extrap = n_partitions / n_trials

    # for each column of mi_array, fit an equation:
    params = np.zeros((3))  # since QE
    # we look at only intercept terms
    if fit == "quadratic":
        params = np.polyfit(x_extrap, mi_array, 2)
        mi_unbiased = params[2]
    elif fit == "linear":
        params = np.polyfit(x_extrap, mi_array, 1)
        mi_unbiased = params[1]

    return mi_unbiased


def pid_unbiased(source_a, source_b, target, fit="quadratic", repeats=5):

    # very similiar to mutual information computation
    n_trials = len(target)
    n_partitions = np.asarray([1, 2, 4])
    pid_array = np.zeros((len(n_partitions), 4))
    pid_unbiased = np.zeros((4))
    # for complete data, i/e, Iplugin
    pid_array[0, :] = pid_plugin(source_a, source_b, target)

    # i'll just do this subsampling once, maybe we should do it more times
    for rxs in range(0, repeats):
        for idx in range(1, len(n_partitions)):

            splits = n_partitions[idx]
            # split data splits the data into equal number of splits
            y_part, x1_part, x2_part = split_data(  # type: ignore
                target=target, sourcea=source_a, sourceb=source_b, splits=splits
            )
            for parts in range(0, splits):
                y_temp = y_part[parts]
                x1_temp = x1_part[parts]
                x2_temp = x2_part[parts]
                # now what
                pid_temp = pid_plugin(source_a=x1_temp, source_b=x2_temp, target=y_temp)
                if splits == 2:
                    pid_array[idx, :] = pid_array[idx, :] + pid_temp / 2
                elif splits == 4:
                    pid_array[idx, :] = pid_array[idx, :] + pid_temp / 4

    # should have 3 pids completely ready
    # now to run polyfit, etc

    pid_array[1, :] = pid_array[1, :] / repeats
    pid_array[2, :] = pid_array[2, :] / repeats

    x_extrap = n_partitions / n_trials

    # for each column of pid_array, fit an equation:
    params = np.zeros((4, 3))  # since QE
    for idx in range(0, 4):
        values = pid_array[:, idx]  # 0 is U1, 1 is U2, 2 is SI and 3 is CI
        if fit == "quadratic":
            params[idx, :] = np.polyfit(x_extrap, values, 2)
            pid_unbiased[idx] = params[idx, 2]  # NOTE: divide here when we do more repetations
        elif fit == "linear":
            params[idx, :] = np.polyfit(x_extrap, values, 1)
            pid_unbiased[idx] = params[idx, 1]  # NOTE: divide here when we do more repetations

    return pid_unbiased


def corrected_mutual_information(source, target, unbiased_measure="quadratic"):

    if unbiased_measure == "plugin":
        return mi_plugin(source, target)
    elif unbiased_measure == "linear":
        return mi_unbiased(source, target, fit="linear")
    elif unbiased_measure == "quadratic":
        return mi_unbiased(source, target, fit="quadratic")


def corrected_pid(sourcea, sourceb, target, unbiased_measure="quadratic"):

    if unbiased_measure == "plugin":
        return pid_plugin(sourcea, sourceb, target)
    elif unbiased_measure == "linear":
        return pid_unbiased(sourcea, sourceb, target, fit="linear")
    elif unbiased_measure == "quadratic":
        return pid_unbiased(sourcea, sourceb, target, fit="quadratic")


def trivariate_plugin(source_a, source_b, target):

    pdf = compute_probability_distribution(target, source_a, source_b)
    trivariate_mi = broja.I_YZ(pdf)

    return trivariate_mi


def correct_trivariate_mi(source_a, source_b, target, repeats=5):

    n_trials = len(target)
    n_partitions = np.asarray([1, 2, 4])
    mi_array = np.zeros((len(n_partitions)))
    mi_unbiased = 0
    # for complete data, i/e, Iplugin
    mi_array[0] = trivariate_plugin(source_a, source_b, target)

    for rxs in range(0, repeats):
        # just one repetation if repeats = 1

        for idx in range(1, len(n_partitions)):
            splits = n_partitions[idx]
            # split data splits the data into equal number of splits
            y_part, x1_part, x2_part = split_data(  # type: ignore
                target=target, sourcea=source_a, sourceb=source_b, splits=splits
            )
            for parts in range(0, splits):
                y_temp = y_part[parts]
                x1_temp = x1_part[parts]
                x2_temp = x2_part[parts]

                # now what
                mi_temp = trivariate_plugin(x1_temp, x2_temp, y_temp)
                if splits == 2:
                    mi_array[idx] = mi_array[idx] + mi_temp / 2
                elif splits == 4:
                    mi_array[idx] = mi_array[idx] + mi_temp / 4

    x_extrap = n_partitions / n_trials
    mi_array[1] = mi_array[1] / repeats
    mi_array[2] = mi_array[2] / repeats

    # for each column of mi_array, fit an equation:
    params = np.zeros((3))  # since QE
    # we look at only intercept terms
    params = np.polyfit(x_extrap, mi_array, 2)
    mi_unbiased = params[2]

    return mi_unbiased


def corrected_tvmi(source_a, source_b, target, unbiased_measure="quadratic"):

    if unbiased_measure == "plugin":
        return trivariate_plugin(source_a, source_b, target)
    elif unbiased_measure == "quadratic":
        return correct_trivariate_mi(source_a, source_b, target)
