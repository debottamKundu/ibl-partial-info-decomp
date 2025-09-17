## try to find significant MI neurons

from one.api import ONE
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from brainbox.ephys_plots import plot_brain_regions
from iblatlas.atlas import AllenAtlas
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from brainwidemap.bwm_loading import merge_probes
from brainbox.behavior.training import compute_performance, plot_psychometric, plot_reaction_time
from brainbox.task.trials import find_trial_ids
from brainbox.io.one import SessionLoader
from pathlib import Path
from brainbox.task.trials import get_event_aligned_raster, get_psth
from brainbox.singlecell import bin_spikes2D
import numpy as np
from iblatlas.atlas import BrainRegions
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
import warnings
from sklearn.ensemble import RandomForestClassifier
from ibl_info.prepare_data_pid import (
    cleaned_regions_flags,
    get_new_cinc_intervals,
    prepare_ephys_data,
)
from ibl_info.selective_decomposition import filter_eids
from ibl_info.utils import (
    alternate_discretize,
    compute_mutual_information,
    compute_pid,
    compute_trivariate_mi,
    FIRING_RATE,
    discretize,
    discretize_keeping_zeros,
    equipopulated_binning,
)
import os
import concurrent.futures
import functools
import random
from ibl_info.utils import check_config
from statsmodels.stats.multitest import multipletests
import ibl_info.measures.information_measures as info
from brainbox.task.closed_loop import generate_pseudo_session

config = check_config()


# compute mi for all neurons, for each eid and regions
# use cluster-id as identifier
# keep all neurons in the dataset


def generate_target(trials_df):
    stim_side = []
    for idx in range(len(trials_df)):
        if trials_df.contrastLeft.iloc[idx] >= 0:
            stim_side.append(0)
        else:
            stim_side.append(1)
    stim_side = np.asarray(stim_side)
    return stim_side


def mi_per_neuron_permuted(spikes, decoding_variable, trials, mask, n_permutations=100):

    mi_observed = info.corrected_mutual_information(  # type: ignore
        source=spikes, target=decoding_variable, unbiased_measure="plugin"
    )
    # use original permutation statistics
    mi_null = np.zeros(n_permutations)
    for i in range(n_permutations):

        # stim_shuffled = np.random.permutation(decoding_variable)
        # mi_null[i] = info.corrected_mutual_information(  # type: ignore
        #     source=spikes, target=stim_shuffled, unbiased_measure="plugin"
        # )
        pseudo_session = generate_pseudo_session(trials)
        pseudo_session = pseudo_session[mask]
        pseudo_targets = generate_target(pseudo_session)
        mi_null[i] = info.corrected_mutual_information(  # type: ignore
            source=spikes, target=pseudo_targets, unbiased_measure="plugin"
        )

    p_value = (np.sum(mi_null >= mi_observed) + 1) / (n_permutations + 1)  # type: ignore
    return mi_observed, p_value


def significant_neurons(spikes, decoding_variable, trials, mask, n_permutations=100, alpha=0.05):

    mi_data = np.zeros((spikes.shape[0]))
    p_values = np.zeros((spikes.shape[0]))
    for idx in range(len(mi_data)):
        mi_data[idx], p_values[idx] = mi_per_neuron_permuted(spikes[idx, :], decoding_variable, trials, mask, n_permutations=n_permutations)  # type: ignore

    # do corrections
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

    return mi_data, p_values, reject


def find_significant_neurons_sessions(session_id, epoch, one, region):

    # not sure if the config is shared
    config = check_config()
    pids, probes = one.eid2pid(session_id)
    if isinstance(probes, list) and len(probes) > 1:
        to_merge = [load_good_units(one, pid=pid, qc=1) for pid in pids]
        spikes, clusters = merge_probes(
            [spikes for spikes, _ in to_merge], [clusters for _, clusters in to_merge]
        )
    else:
        spikes, clusters = load_good_units(one, pid=pids[0], qc=1)

    trials, mask = load_trials_and_mask(
        one, session_id, exclude_nochoice=True, exclude_unbiased=True
    )

    trials_masked = trials[mask]

    intervals, target_variable, congruent_flags, incongruent_flags = get_new_cinc_intervals(
        trials_masked, epoch
    )

    binned_spikes, actual_regions, n_units, cluster_uuids_list = prepare_ephys_data(
        spikes, clusters, intervals, [region], minimum_units=1
    )

    spike_data = binned_spikes[0].T
    discretize_method = config["discretize"]
    n_bins = config["n_bins"]
    if discretize_method == 1:
        discretized_spikes = alternate_discretize(spike_data, n_bins=n_bins)
    else:
        # we can also do equipopulated
        discretized_spikes = discretize_keeping_zeros(spike_data, n_bins=n_bins)

    mi_data, p_values, reject = significant_neurons(
        discretized_spikes, target_variable, trials, mask, n_permutations=1000, alpha=0.05
    )

    information_pkl = {}
    information_pkl["mi_data"] = mi_data
    information_pkl["p_values"] = p_values
    information_pkl["reject"] = reject
    information_pkl["uuids"] = np.asarray(cluster_uuids_list[0])  # probably

    return information_pkl


def run_checks(task_tuple):

    eid, region, epoch = task_tuple
    one = ONE()
    try:
        # ideally information pickle, but i want to subsample mutliple times
        information_pickle = find_significant_neurons_sessions(eid, epoch, one, region)
        if information_pickle == {}:
            return region, eid, None
        else:
            return region, eid, information_pickle
    except Exception as e:
        print(f"Error regarding {eid} in region {region}: {e}")
        return region, eid, None


def run_flattened(list_of_regions, epoch):

    one = ONE()
    unit_df = bwm_units(one)
    all_tasks_to_run = []
    for region in list_of_regions:
        selective_eids = filter_eids(unit_df, region, significant_filter=config["decoder_filter"])
        for eid in tqdm(selective_eids):
            all_tasks_to_run.append((eid, region, epoch))

    print(f"Total tasks: {len(all_tasks_to_run)}")

    processed_results = []
    workers = os.cpu_count() // 2  # type: ignore
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results_iterator = executor.map(run_checks, all_tasks_to_run)

        processed_results = list(
            tqdm(results_iterator, total=len(all_tasks_to_run), desc="Processing Tasks")
        )

    print("Now collecting and writing out")

    eid_data = {}
    for region, eid, information_pickle in processed_results:

        if eid not in eid_data:
            eid_data[eid] = {}
            eid_data[eid]["config"] = config  # lot of repeats but atleast stores the config

        if information_pickle is not None:
            eid_data[eid][region] = information_pickle

    # if config["decoder_filter"]:
    #     suffix = "_better_sessions"
    # else:
    #     suffix = "_all_sessions"

    # if config["discretize"] == 1:
    #     suffix += "_alternate"
    # else:
    #     suffix += "_equipopulated"

    # n_bins = config["n_bins"]
    # suffix += f"_{n_bins}"

    # this will make one huge pickle:
    for eid, eid_pickle in eid_data.items():
        with open(f"./data/generated/mi_significant_neurons_pseudo_{eid}_{epoch}.pkl", "wb") as f:
            pkl.dump(eid_pickle, f)

    print("Done!")


if __name__ == "__main__":
    important_regions = config["stim_prior_regions"]
    run_flattened(important_regions, "stim")
