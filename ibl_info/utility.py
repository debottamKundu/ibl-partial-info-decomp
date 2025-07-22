# utility functions
# - aggregating clusters into regions
# - helpers for plotting results
# - computing PID for a target and 2 sources
# - computing net synergy/redundancy for one target and two sources
# - computing just MI


from one.api import ONE
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from brainbox.ephys_plots import plot_brain_regions
from brainbox.task.trials import get_event_aligned_raster, get_psth
from iblatlas.atlas import AllenAtlas
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from brainbox.behavior.training import compute_performance, plot_psychometric, plot_reaction_time
from brainbox.task.trials import find_trial_ids
from brainbox.io.one import SessionLoader
from pathlib import Path
from brainbox.task.trials import get_event_aligned_raster, get_psth
from brainbox import singlecell
import numpy as np
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
import pandas as pd
import itertools
from decision_rnn.information_measures.processing import equipopulated_binning
import ibl_info.measures.information_measures as info


def aggregated_regions_time_resolved(binned_spike_counts, cluster_acronyms):
    """generates summed over spike counts for each region, with a time resolution provided beforehand


    Args:
        binned_spike_counts (np.array): trials x neurons x timepoints
        cluster_acronyms (np.array): neurons

    Returns:
        np.array: binned counts aggregated by regions, and names of regions
    """
    # binned_spike_counts = trials x neurons x timepoints

    regions = np.unique(cluster_acronyms)
    data = np.zeros(
        (binned_spike_counts.shape[0], len(regions), binned_spike_counts.shape[-1])
    )  # trials, regions, timepoints
    for idx, r in enumerate(regions):
        neurons = np.argwhere(cluster_acronyms == r).reshape(
            -1,
        )
        aggregate_cluster = np.sum(binned_spike_counts[:, neurons, :], axis=1)
        data[:, idx, :] = aggregate_cluster
    return data, regions


def maintain_neural_count(neural_data, regions, minimum_number=10):
    """
    Ensure that the total number of neurons in each region is greater than a specified minimum number

    Args:
        neural_data (np.array): neurons x trials
        regions (np.array): name of regions for each neuron
        minimum_number (int) : minimum number of neurons in each region required to pass, defaults to 5
    """

    unique_regions, neuron_per_region = np.unique(regions, return_counts=True)
    valid_regions_idx = np.argwhere(neuron_per_region > minimum_number)
    valid_regions = unique_regions[valid_regions_idx].reshape(
        -1,
    )

    # now to get only data from valid regions
    region_idx = np.argwhere(np.isin(regions, valid_regions)).reshape(
        -1,
    )
    neural_data = neural_data[region_idx, :]
    regions = regions[region_idx].reshape(
        -1,
    )

    return neural_data, regions


def aggregated_regions_time_intervals(spike_counts, cluster_acronyms, average=False):
    """generates total number of spikes per region for different trials

    Args:
        spike_counts (np.array): spike count for each trial from each neuron, # trials x neurons
        cluster_acronyms (np.array): neuron locations
    Returns:
        np.array: spike count aggregated by regions, and names of  corresponding regions
    """

    # remember that here the data structure for spike counts is trials x neurons
    spike_counts, cluster_acronyms = maintain_neural_count(spike_counts.T, cluster_acronyms)
    spike_counts = spike_counts.T  # flip it back into trials x neurons

    regions = np.unique(cluster_acronyms)
    data = np.zeros((spike_counts.shape[0], len(regions)))  # trials x regions
    for idx, r in enumerate(regions):
        neurons = np.argwhere(cluster_acronyms == r).reshape(
            -1,
        )
        if average:
            aggregate_cluster = np.sum(spike_counts[:, neurons], axis=1) // len(neurons)
        else:
            aggregate_cluster = np.sum(spike_counts[:, neurons], axis=1)
        data[:, idx] = aggregate_cluster
    return data, regions


def discretize_neural_data(neural_data, method="neuron", n_bins=4):
    """
    Discretize the spike counts into equipopulated bins

    Args:
        neural_data (np.array): spike counts for neurons x trials
        method (str, optional): how to determine the percenille.
                                Defaults to 'neuron'. Calculate the percentile per neuron
                                Other options: 'all': Calculate the percentile based on the entire dataset
    """
    print(neural_data.shape, method)
    if method == "neuron":
        discrete_data = np.zeros((neural_data.shape[0], neural_data.shape[1]))
        # discretize per recorded neuron
        for idx in tqdm(range(neural_data.shape[0])):

            row = neural_data[idx, :]
            # bin_edges = np.percentile(row, [20,40,60,80])
            # set bin edges to 4 parts
            # bin_edges = np.percentile(row, [25,50,75])
            # discrete_data[idx, :] = np.digitize(row, bin_edges)
            discrete_row, bin_edges_p = pd.qcut(
                row, q=n_bins, labels=False, duplicates="drop", retbins=True
            )
            discrete_data[idx, :] = discrete_row
    elif method == "all":
        bin_edges = np.percentile(neural_data, [20, 40, 60, 80])
        # set bin edges to 4 parts
        # bin_edges = np.percentile(neural_data, [25,50,75])
        discrete_data = np.digitize(neural_data, bin_edges)
    elif method == "none":
        return neural_data
    else:
        raise NotImplementedError
    discrete_data = np.nan_to_num(discrete_data, nan=0)
    return discrete_data


def subsample(neural_data, decoding_variable, percentage=0.75):
    """
    Subsample a portion of the trials

    Args:
        neural_data (np.array): neurons x trials
        percentage (float, optional): percentage of trials to subsample. Defaults to .75.
    """

    total_trials = neural_data.shape[1]
    subsampled = int(total_trials * percentage)
    trials_sampled = np.random.choice(np.arange(0, total_trials), subsampled, replace=False)
    neural_data = neural_data[:, np.sort(trials_sampled).astype(np.int16)]
    decoding_variable = decoding_variable[np.sort(trials_sampled).astype(np.int16)]
    return neural_data, decoding_variable


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
