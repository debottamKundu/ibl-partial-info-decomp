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
    get_window,
    prepare_ephys_data,
    get_congruent_incongruent_intervals,
    cleaned_regions_single_region,
)
from ibl_info.utility import alternate_discretize, compute_mutual_information, compute_pid
import os
import concurrent.futures
import functools


def run_analysis_single_condition(spikes, clusters, intervals, region, target_variable):

    binned_spikes, actual_regions, n_units, cluster_uuids_list = prepare_ephys_data(
        spikes, clusters, intervals, [region], minimum_units=10
    )  # this returns all neurons from a single region that pass qc
    # however, it is in trials x neurons
    # i flip it

    # check if anything is ever returned
    if len(binned_spikes) == 0:
        # return empty arrays
        return np.asarray([]), np.asarray([])

    spike_data = binned_spikes[0].T
    # clean this up ; throw away non-responsive neurons
    cleaned_binned_spikes = cleaned_regions_single_region(
        spike_data, percent_of_no_spikes_threshold=0.3
    )

    # use the alternate binning
    # NOTE: reduced binning here
    discretized_spikes = alternate_discretize(cleaned_binned_spikes, n_bins=2)  # only 5 bins

    mutual_information = compute_mutual_information(discretized_spikes, target_variable)
    pid = compute_pid(data=discretized_spikes, targets=target_variable)

    return mutual_information, pid


def prepare_neural_data(session_id, epoch, one, region):

    pids, probes = one.eid2pid(session_id)
    if isinstance(probes, list) and len(probes) > 1:
        to_merge = [load_good_units(one, pid=pid, qc=1) for pid in pids]
        spikes, clusters = merge_probes(
            [spikes for spikes, _ in to_merge], [clusters for _, clusters in to_merge]
        )
    else:
        spikes, clusters = load_good_units(one, pid=pids[0], qc=1)

    # i only want one region normally
    # or maybe we check how many regions this animal has
    # that makes sense

    window = get_window(epoch)
    print(window)

    trials, mask = load_trials_and_mask(
        one, session_id, exclude_nochoice=True, exclude_unbiased=True
    )
    trials = trials[mask]

    # for now we are looking at just (stimulus interval)
    # we know the order
    labels = ["all", "congruent", "incongruent", "middling_incongruent"]
    intervals, decoding_variables = get_congruent_incongruent_intervals(trials, epoch)

    information_pickle = {}

    for idx in range(len(intervals)):
        interval = intervals[idx]
        decoding_variable = decoding_variables[idx]
        print(f"Running analysis for {epoch} - {region} - {labels[idx]}")
        mutual_information, pid = run_analysis_single_condition(
            spikes, clusters, interval, region, decoding_variable
        )

        information_pickle[labels[idx]] = {
            "mutual_information": mutual_information,
            "pid": pid,
        }

    return information_pickle


def filter_eids(unit_df, region):
    unit_df_region = unit_df[unit_df["Beryl"] == region]
    eids = np.unique(unit_df_region["eid"])
    return eids


def trials_used(session_id, epoch, one, region):

    # i only want one region normally
    # or maybe we check how many regions this animal has
    # that makes sense
    window = get_window(epoch)
    print(window)

    trials, mask = load_trials_and_mask(
        one, session_id, exclude_nochoice=True, exclude_unbiased=True
    )
    trials = trials[mask]

    # for now we are looking at just (stimulus interval)
    # we know the order
    labels = ["all", "congruent", "incongruent", "middling_incongruent"]
    intervals, decoding_variables = get_congruent_incongruent_intervals(trials, epoch)

    trials_used = np.zeros((4))
    for idx in range(len(intervals)):
        trials_used[idx] = decoding_variables[idx].shape[0]

    return trials_used


def run_selective_decomposition(one, list_of_regions, epoch):

    unit_df = bwm_units(one)
    for region in list_of_regions:

        selective_eids = filter_eids(unit_df, region)
        region_pickle = {}
        for eid in tqdm(selective_eids):

            # if eids_done >= 2:
            #     break
            try:
                information_pickle = prepare_neural_data(eid, epoch, one, region)
                region_pickle[eid] = information_pickle
                # eids_done += 1
            except Exception as e:
                print(e)
                continue
        with open(f"./data/generated/selective_decomposition_{region}_{epoch}.pkl", "wb") as f:
            pkl.dump(region_pickle, f)


def process_trials_used(region, epoch):

    one = ONE()
    unit_df = bwm_units(one)
    selective_eids = filter_eids(unit_df, region)
    region_pickle = {}
    for eid in tqdm(selective_eids):
        try:
            trials_used = trials_used(eid, epoch, one, region)
            region_pickle[eid] = trials_used
        except Exception as e:
            print(e)
            continue

    with open(f"./data/generated/trials_used_{region}_{epoch}.pkl", "wb") as f:
        pkl.dump(region_pickle, f)

    print(f"Worker {os.getpid()} finished {region}")
    return region


# refactoring so that i can run this in parallel
def process_region_task(region, epoch):

    one = ONE()
    unit_df = bwm_units(one)
    selective_eids = filter_eids(unit_df, region)
    region_pickle = {}
    for eid in tqdm(selective_eids):
        try:
            information_pickle = prepare_neural_data(eid, epoch, one, region)
            region_pickle[eid] = information_pickle
        except Exception as e:
            print(e)
            continue

    with open(f"./data/generated/selective_decomposition_{region}_{epoch}.pkl", "wb") as f:
        pkl.dump(region_pickle, f)

    print(f"Worker {os.getpid()} finished {region}")
    return region


def run_selective_decomposition_parallel(list_of_regions, epoch):

    max_regions = len(list_of_regions)
    partial_process_region = functools.partial(
        process_region_task,
        epoch=epoch,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_regions) as executor:

        results_iterator = executor.map(partial_process_region, list_of_regions)

        for result in tqdm(
            results_iterator, total=len(list_of_regions), desc="Processing Regions"
        ):
            print(result)


def run_selective_trial_collation(list_of_regions, epoch):

    max_regions = len(list_of_regions)
    partial_process_trials = functools.partial(
        process_trials_used,
        epoch=epoch,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_regions) as executor:

        results_iterator = executor.map(partial_process_trials, list_of_regions)

        for result in tqdm(
            results_iterator, total=len(list_of_regions), desc="Processing Regions"
        ):
            print(result)


if __name__ == "__main__":

    important_regions = [
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

    # one = ONE()
    # run_selective_decomposition(one, important_regions, "stim")

    # run_selective_decomposition_parallel(important_regions, "stim")
    run_selective_trial_collation(important_regions, "stim")

    # three random regions; one that has only stim but no prior; one prior but no stim, one just choice
    random_regions = ["SCs", "VISa", "PO"]

    run_selective_decomposition_parallel(random_regions, "stim")
    run_selective_trial_collation(random_regions, "stim")
