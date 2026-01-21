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
    get_new_cinc_intervals_choice,
    prepare_ephys_data,
    return_significant_cells,
)
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

config = check_config()

# define const
PERCENT_OF_SPIKE_THRESHOLD = 0.4


def select_neurons_for_analysis_all(spikes, clusters, intervals, region, session_id=None):

    binned_spikes, actual_regions, n_units, cluster_uuids_list = prepare_ephys_data(
        spikes, clusters, intervals, [region], minimum_units=3
    )

    if config["discretize"] == 2:
        addendum = "equi"
    else:
        addendum = "_alternate"

    n_bins = config["n_bins"]

    if session_id is not None:

        if config["single_cell_filter"]:

            if not config["mi_filter"]:
                filename = f"./data/processed/singlecellsmallwindow//significance_results_{session_id}_smallwindow.csv"
                df = pd.read_csv(filename)
                ccids_significant = df[df["p_value"] <= 0.05]["QC_cluster_id"].values
                mask = np.isin(np.asarray(cluster_uuids_list[0]), ccids_significant)  # type: ignore
            elif config["mi_filter"]:

                print("MI filter applied")
                # change this here
                if config["epoch"] == "stim":
                    filename = f"./data/generated/cellmi/pseudo_new/mi_significant_neurons_properpseudo_{session_id}_stim_{addendum}_{n_bins}.pkl"
                elif config["epoch"] == "choice":
                    # filename = f"./data/generated/choice/choicesignificance/mi_significant_neurons_choice_{session_id}_choice.pkl"
                    # use new filename
                    addendum = ""
                    if config["discretize"] == 2:
                        addendum = "_equi"
                    # TODO: find a cleaner solution for the filename
                    filename = f"./data/generated/choice/significantneurons/choicesignificance_pseudo_{addendum}/mi_significant_neurons_properpseudo_{session_id}_choice.pkl"
                with open(filename, "rb") as f:
                    mi_data = pkl.load(f)
                mi_data_region = mi_data[region]
                ccids = mi_data_region["uuids"]
                significant = mi_data_region["reject"]
                mask = np.isin(np.asarray(cluster_uuids_list[0]), np.asarray(ccids)[significant])
        else:
            print("No filter applied: shouldn't happen")
            mask = np.ones(len(cluster_uuids_list[0]), dtype=bool)
    else:
        mask = np.ones(len(cluster_uuids_list[0]), dtype=bool)
        # essentially status quo

    if len(binned_spikes) == 0:
        # return empty arrays
        return [0]  # no neurons viable

    spike_data = binned_spikes[0].T
    if session_id is not None:
        firing_rate_threshold = 0
        percent_spike = 0
    else:
        firing_rate_threshold = FIRING_RATE[region]
        percent_spike = PERCENT_OF_SPIKE_THRESHOLD

    cleaned_neurons = cleaned_regions_flags(
        spike_data,
        firing_rate_threshold=firing_rate_threshold,
        percent_of_no_spikes_threshold=percent_spike,
    )

    cleaned_neurons = cleaned_neurons & mask  # so that it all aligns

    return cleaned_neurons


def compute_condition(target, spikes):
    mutual_information = compute_mutual_information(spikes, target)
    pid = compute_pid(data=spikes, targets=target)
    trivariate = compute_trivariate_mi(data=spikes, targets=target)

    return {
        "mutual_information": mutual_information,
        "pid": pid,
        "trivariate": trivariate,
    }


def compute_subsampled(congruent_spikes, congruent_targets, incongruent_targets):

    left_fraction = np.sum(incongruent_targets == 1) / len(incongruent_targets)

    # we want to ensure similar fraction for congruent subsampling
    left_congruent = np.where(congruent_targets == 1)[0]
    right_congruent = np.where(congruent_targets == 0)[0]

    sampled_mi = []
    sampled_pid = []
    sampled_joint = []
    for repeats in range(5):  # should be 5 or more, lower in order to speed up

        n_left_subsample = int(np.round(left_fraction * len(incongruent_targets)))
        n_right_subsample = int(len(incongruent_targets) - n_left_subsample)

        # now we need to do the actual subsampling
        selected_indices_left = np.random.choice(left_congruent, n_left_subsample, replace=False)
        selected_indices_right = np.random.choice(
            right_congruent, n_right_subsample, replace=False
        )

        selected_indices = np.concatenate((selected_indices_left, selected_indices_right))
        subsampled_targets = congruent_targets[selected_indices]
        subsampled_spikes = congruent_spikes[:, selected_indices]

        info_ = compute_condition(subsampled_targets, subsampled_spikes)
        sampled_mi.append(info_["mutual_information"])
        sampled_pid.append(info_["pid"])
        sampled_joint.append(info_["trivariate"])

    # average
    sampled_mi = np.asarray(sampled_mi)
    sampled_pid = np.asarray(sampled_pid)
    sampled_joint = np.asarray(sampled_joint)

    sampled_mi = np.mean(sampled_mi, axis=0)
    sampled_pid = np.mean(sampled_pid, axis=0)
    sampled_joint = np.mean(sampled_joint, axis=0)

    return {
        "mutual_information": sampled_mi,
        "pid": sampled_pid,
        "trivariate": sampled_joint,
    }


def run_analysis_single_session(
    session_id, epoch, one, region, discretize_method=1, single_cell_filter=False
):

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
    trials = trials[mask]

    if epoch == "stim":
        intervals, target_variable, congruent_flags, incongruent_flags = get_new_cinc_intervals(
            trials, epoch
        )
    elif epoch == "choice":
        intervals, target_variable, congruent_flags, incongruent_flags = (
            get_new_cinc_intervals_choice(trials, epoch)
        )
    else:
        raise NotImplementedError

    trial_count = np.zeros((3))

    trial_count[0] = intervals.shape[0]
    trial_count[1] = np.sum(congruent_flags)
    trial_count[2] = np.sum(incongruent_flags)

    information_pickle = {}

    # find the neurons to be used:
    # for all

    binned_spikes, actual_regions, n_units, cluster_uuids_list = prepare_ephys_data(
        spikes, clusters, intervals, [region], minimum_units=3
    )  # this returns all neurons from a single region that pass qc
    # however, it is in trials x neurons
    if single_cell_filter:
        # use new function with bwm single cells
        # neuron_flags = return_significant_cells(session_id, epoch, cluster_uuids_list)
        # use old significant neurons
        neuron_flags = select_neurons_for_analysis_all(
            spikes, clusters, intervals, region, session_id=session_id
        )
    else:
        neuron_flags = select_neurons_for_analysis_all(spikes, clusters, intervals, region)
    if np.sum(neuron_flags) < 2:
        return information_pickle

    # we want neurons x trials
    spike_data = binned_spikes[0].T

    # clean up with neuron flags
    spike_data = spike_data[neuron_flags, :]

    # discretize here
    n_bins = config["n_bins"]
    if discretize_method == 1:
        discretized_spikes = alternate_discretize(spike_data, n_bins=n_bins)
    else:
        # we can also do equipopulated
        discretized_spikes = discretize_keeping_zeros(spike_data, n_bins=n_bins)

    information_pickle["trials"] = trial_count
    information_pickle["neurons"] = np.sum(neuron_flags)
    information_pickle["average_spikes"] = np.mean(spike_data, axis=1)

    information_pickle["all"] = compute_condition(target_variable, discretized_spikes)
    # congruent trials
    congruent_spikes = discretized_spikes[:, congruent_flags]
    congruent_targets = target_variable[congruent_flags]
    information_pickle["congruent"] = compute_condition(congruent_targets, congruent_spikes)
    # incongruent trials
    incongruent_spikes = discretized_spikes[:, incongruent_flags]
    incongruent_targets = target_variable[incongruent_flags]

    information_pickle["incongruent"] = compute_condition(incongruent_targets, incongruent_spikes)

    # compute subsamples on
    information_pickle["subsampled"] = compute_subsampled(
        congruent_spikes, congruent_targets, incongruent_targets
    )

    return information_pickle


def filter_eids(unit_df, region, significant_filter=False):
    unit_df_region = unit_df[unit_df["Beryl"] == region]
    eids = np.unique(unit_df_region["eid"])

    # here we can filter on significant eids
    if significant_filter:
        df_stim_decoder = pd.read_parquet(config["pq_location"])
        df_stim_decoder = df_stim_decoder[df_stim_decoder["region"] == region]
        significant_dfs = df_stim_decoder[df_stim_decoder["p-value"] < 0.05]
        eids = np.intersect1d(eids, significant_dfs["eid"].values)

    return eids


def run_selective_decomposition(one, list_of_regions, epoch):

    unit_df = bwm_units(one)
    for region in list_of_regions:

        selective_eids = filter_eids(unit_df, region)
        region_pickle = {}
        for eid in tqdm(selective_eids):
            try:
                information_pickle = run_analysis_single_session(eid, epoch, one, region)
                region_pickle[eid] = information_pickle
                # eids_done += 1
            except Exception as e:
                print(e)
                continue
        with open(f"./data/generated/selective_decomposition_{region}_{epoch}.pkl", "wb") as f:
            pkl.dump(region_pickle, f)


if __name__ == "__main__":

    important_regions = np.asarray(
        [
            "VISp",
            "MOs",
            "SSp-ul",
            "ACAd",
            "PL",
            "CP",
            "VPM",
            "MG",
            "LGd",
            "ZI",
            "SNr",
            "MRN",
            "SCm",
            "PAG",
            "APN",
            "RN",
            "PPN",
            "PRNc",
            "PRNr",
            "GRN",
            "IRN",
            "PGRN",
            "CUL4 5",
            "SIM",
            "IP",
        ]
    )

    one = ONE()
    subject_id = "CSH_ZAD_022"
    eid = "a82800ce-f4e3-4464-9b80-4c3d6fade333"
    session_id = eid
    ipickle_singlee = run_analysis_single_session(
        session_id=session_id,
        one=one,
        region="LGd",
        discretize_method=1,
        epoch="stim",
        single_cell_filter=config["single_cell_filter"],
    )
    with open(f"./data/generated/test_{subject_id}_{session_id}.pkl", "wb") as f:
        pkl.dump(ipickle_singlee, f)

    # run_selective_decomposition(one, important_regions[0:2], "stim")

    # run_selective_decomposition_parallel(important_regions, "stim")

    # # three random regions; one that has only stim but no prior; one prior but no stim, one just choice
    # random_regions = ["SCs", "VISa", "PO"]
    # run_selective_decomposition_parallel(random_regions, "stim")
