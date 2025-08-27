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
    get_window,
    prepare_ephys_data,
    get_congruent_incongruent_intervals,
)
from ibl_info.selective_decomposition import filter_eids
from ibl_info.utils import (
    alternate_discretize,
    compute_mutual_information,
    compute_pid,
    compute_trivariate_mi,
    FIRING_RATE,
    discretize,
    equipopulated_binning,
)
import os
import concurrent.futures
import functools
import random


def run_for_session(session_id, epoch, one, region, df):

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

    intervals, target_variable, congruent_flags, incongruent_flags = get_new_cinc_intervals(
        trials, epoch
    )

    binned_spikes, actual_regions, n_units, cluster_uuids_list = prepare_ephys_data(
        spikes, clusters, intervals, [region], minimum_units=1
    )

    ccids_significant = df[df["p_value"] <= 0.05]["QC_cluster_id"].values

    nxs = len(
        list(set(cluster_uuids_list[0]).intersection(set(ccids_significant)))
    )  # get significant neurons

    return nxs


def check_single_cell_statistics(list_of_regions, epoch):

    one = ONE()
    unit_df = bwm_units(one)
    all_regions = {}
    for region in list_of_regions:

        selective_eids = filter_eids(unit_df, region)
        region_pickle = {}
        for eid in tqdm(selective_eids):
            try:
                # we get cluster_id_list here from requisite data frame
                filename = f"./data/processed/singlecellresults/significance_results_{eid}.csv"
                df = pd.read_csv(filename)
                number_of_neurons = run_for_session(eid, epoch, one, region, df)
                region_pickle[eid] = number_of_neurons
            except Exception as e:
                print(e)
                continue

        all_regions[region] = region_pickle

    with open(f"./data/generated/single_cell_test_passed_{epoch}.pkl", "wb") as f:
        pkl.dump(all_regions, f)


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

    check_single_cell_statistics(important_regions, "stim")
