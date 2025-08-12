import itertools
from pathlib import Path

import ibl_info.measures.information_measures as info
import numpy as np
import yaml
from brainbox.io.one import SessionLoader
from brainwidemap import load_good_units
import pandas as pd
from tqdm import tqdm


def equipopulated_binning(signal, n_bins=5):
    """
    Discretize the hidden state into equipopulated bins

    Args:
        signal (np.array): signal, trialsx1
        n_bins (int): number of bins

    """

    discrete_data = np.zeros_like(signal)
    # discretize per recorded neuron

    discrete_row, bin_edges_p = pd.qcut(
        signal, q=n_bins, labels=False, duplicates="drop", retbins=True
    )
    discrete_data = discrete_row

    discrete_data = np.nan_to_num(discrete_data, nan=0)
    return discrete_data


def download_data(one, pid):
    spikes, clusters = load_good_units(one, pid, compute_metrics=False)
    return spikes, clusters


def generate_source_ids(number_of_neurons):
    combinations_neuronids = []
    for x in itertools.combinations(range(number_of_neurons), 2):
        combinations_neuronids.append([x[0], x[1]])

    combinations_neuronids = np.asarray(combinations_neuronids)
    return combinations_neuronids


def discretize(spike_data, n_bins=5):
    """discretize into specified number of equipopulated bins

    Args:
        spike_data (np.array): neurons x trials
    """

    discrete_data = np.zeros_like(spike_data)
    for neurons in range(spike_data.shape[0]):
        discrete_data[neurons, :] = equipopulated_binning(spike_data[neurons, :], n_bins=n_bins)
    return discrete_data


def alternate_discretize(spike_data, n_bins=3):
    """
    if the neurons don't fire enough, maybe it makes more sense to just round down the greater>5 ones into a variable
    """

    n_bins = n_bins - 1  # to account for 0
    # if 3, now 0:0, 1:1, >=2:2
    discrete_data = np.zeros_like(spike_data)
    spike_data = spike_data.copy()  # deep copy
    for neurons in range(spike_data.shape[0]):

        A = spike_data[neurons, :]
        A[A >= n_bins] = n_bins

        discrete_data[neurons, :] = A

    return discrete_data


def compute_mutual_information(neural_data, decoding_variable):
    mi_data = np.zeros((neural_data.shape[0]))
    for idx in range(len(mi_data)):
        mi_data[idx] = info.corrected_mutual_information(  # type: ignore
            source=neural_data[idx, :], target=decoding_variable, unbiased_measure="quadratic"
        )
    return mi_data


def compute_pid(data, targets, unbiased_measure="quadratic"):

    sources = generate_source_ids(data.shape[0])
    pid_information = np.zeros((len(sources), 4))  # neuronsC2 x 4
    for idx in tqdm(
        range(len(sources)), desc="Running for all sources", leave=False
    ):  # this is the place to introduce parallelization
        s1 = sources[idx][0]
        s2 = sources[idx][1]
        X1 = np.asarray(data[s1, :], dtype=np.int32)
        X2 = np.asarray(data[s2, :], dtype=np.int32)
        Y = np.asarray(targets, dtype=np.int32)
        u1, u2, red, syn = info.corrected_pid(sourcea=X1, sourceb=X2, target=Y, unbiased_measure=unbiased_measure)  # type: ignore
        pid_information[idx, :] = u1, u2, red, syn

    return pid_information


def compute_trivariate_mi(data, targets):

    sources = generate_source_ids(data.shape[0])
    trivariate_information = np.zeros((len(sources), 1))  # neuronsC2 x 4
    for idx in tqdm(
        range(len(sources)), desc="Running for all sources", leave=False
    ):  # this is the place to introduce parallelization
        s1 = sources[idx][0]
        s2 = sources[idx][1]
        X1 = np.asarray(data[s1, :], dtype=np.int32)
        X2 = np.asarray(data[s2, :], dtype=np.int32)
        Y = np.asarray(targets, dtype=np.int32)

        trivariate_information[idx] = info.corrected_tvmi(source_a=X1, source_b=X2, target=Y)  # type: ignore

    return trivariate_information


def check_config():
    """Load config yaml and perform some basic checks"""
    # Get config
    with open(Path(__file__).parent.parent.joinpath("config.yaml"), "r") as config_yml:
        config = yaml.safe_load(config_yml)
    return config


def epoch_events(epoch):
    if epoch == "stim":
        return "stimOn_times"
    else:
        raise NotImplementedError


# define constants

FIRING_RATE = {
    np.str_("VISp"): np.float64(1.2),
    np.str_("MOs"): np.float64(0.9057971014492754),
    np.str_("SSp-ul"): np.float64(1.1545157621000042),
    np.str_("ACAd"): np.float64(1.1401425178147269),
    np.str_("PL"): np.float64(0.5133919464730121),
    np.str_("CP"): np.float64(0.778816199376947),
    np.str_("VPM"): np.float64(3.084529679795269),
    np.str_("MG"): np.float64(2.4375),
    np.str_("LGd"): np.float64(3.461941533370105),
    np.str_("ZI"): np.float64(2.7667320801487296),
    np.str_("SNr"): np.float64(6.982774502579218),
    np.str_("MRN"): np.float64(2.4242454513972134),
    np.str_("SCm"): np.float64(1.8298969072164948),
    np.str_("PAG"): np.float64(1.3695090439276485),
    np.str_("APN"): np.float64(2.7202380952380953),
    np.str_("RN"): np.float64(2.197309417040359),
    np.str_("PPN"): np.float64(0.90007215007215),
    np.str_("PRNc"): np.float64(1.7209121557497808),
    np.str_("PRNr"): np.float64(1.2751677852348993),
    np.str_("GRN"): np.float64(1.9773492500765228),
    np.str_("IRN"): np.float64(0.9086647160526665),
    np.str_("PGRN"): np.float64(1.1100909703504043),
    np.str_("CUL4 5"): np.float64(1.6),
    np.str_("SIM"): np.float64(2.1323529411764706),
    np.str_("IP"): np.float64(5.979498861047836),
}
