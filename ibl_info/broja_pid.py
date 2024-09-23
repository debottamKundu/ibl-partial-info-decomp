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
    return [
        info_decomposition["UIY"],
        info_decomposition["UIZ"],
        info_decomposition["SI"],
        info_decomposition["CI"],
    ]

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
