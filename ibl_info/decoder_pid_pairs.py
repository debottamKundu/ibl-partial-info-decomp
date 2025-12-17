# load neuron pairs and do the same thing
# imo the decoding function signature stays the same
# we just need to figure out the loading
# so for each session
# check what pairs exist
# and then compute

import concurrent.futures
import functools
import itertools
import os
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from brainbox.behavior.training import compute_performance, plot_psychometric, plot_reaction_time
from brainbox.ephys_plots import plot_brain_regions
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.singlecell import bin_spikes2D
from brainbox.task.trials import find_trial_ids, get_event_aligned_raster, get_psth
from brainwidemap import bwm_query, bwm_units, load_good_units, load_trials_and_mask
from brainwidemap.bwm_loading import merge_probes
from iblatlas.atlas import AllenAtlas, BrainRegions
from matplotlib import pyplot as plt
from one.api import ONE
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import compute_sample_weight
from tqdm import tqdm

from ibl_info.dual_decoders import compute_null_distribution
import ibl_info.measures.information_measures as info
from ibl_info.decoder_pid import compute_decoder_pid
from ibl_info.decoder_pid_wifi import region_combinations
from ibl_info.prepare_data_pid import (
    get_new_cinc_intervals,
    get_new_cinc_intervals_choice,
    prepare_ephys_data,
)
from ibl_info.utils import check_config, equipopulated_binning, equispaced_binning

config = check_config()


def load_region(spikes, clusters, intervals, region):

    binned_spikes, actual_regions, n_units, cluster_uuids_list = prepare_ephys_data(
        spikes, clusters, intervals, [region], minimum_units=config["min_units_decoding"]
    )  # this returns all neurons from a single region that pass qc

    if len(binned_spikes) == 0:
        print(f'Neurons less than {config["min_units_decoding"]} in {region}')
        return []
    else:
        return binned_spikes[0]


def run_pair_pid(session_id, region_a, region_b, epoch):
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

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

    spike_data_a = load_region(spikes, clusters, intervals, region_a)
    spike_data_b = load_region(spikes, clusters, intervals, region_b)

    if spike_data_a == [] or spike_data_b == []:
        return {}

    trial_count = np.zeros((3))

    trial_count[0] = intervals.shape[0]
    trial_count[1] = np.sum(congruent_flags)
    trial_count[2] = np.sum(incongruent_flags)

    # now we add the new decoder functions
    information_pickle = {}
    information_pickle["neurons"] = [len(spike_data_a), len(spike_data_b)]
    information_pickle["trials"] = trial_count

    information_results, results = compute_decoder_pid(
        target=target_variable,
        spikes_a=spike_data_a,
        spikes_b=spike_data_b,
        n_bootstraps=config["n_bootstraps_decoding"],
        n_bins=config["n_bins_decoding"],
        congruent_mask=congruent_flags,
        incongruent_mask=incongruent_flags,
        decoder_output_only=config["decoder_output_only"],
    )
    null_results = compute_null_distribution(
        neural_data_A=spike_data_a,
        neural_data_B=spike_data_b,
        trial_labels=target_variable,
        subset_size_D=10,
        n_permutations=50,
        n_splits=5,
    )

    information_pickle["information"] = information_results
    information_pickle["decoding_results"] = results
    information_pickle["null_results"] = null_results

    return information_pickle


def prepare_and_run_data(task_tuple):

    eid, region_a, region_b, epoch = task_tuple

    try:
        # ideally information pickle, but i want to subsample mutliple times
        information_pickle = run_pair_pid(eid, region_a, region_b, epoch)
        if information_pickle == {}:
            return f"{region_a}-{region_b}", eid, None
        else:
            return f"{region_a}-{region_b}", eid, information_pickle
    except Exception as e:
        print(f"Error regarding {eid} in region {[region_a,region_b]}: {e}")
        return f"{region_a}-{region_b}", eid, None


def filter_eids_for_region_pair(unit_df, region_a, region_b):
    unit_df_region_a = unit_df[unit_df["Beryl"] == region_a]
    unit_df_region_b = unit_df[unit_df["Beryl"] == region_b]
    eids_a = np.unique(unit_df_region_a["eid"])
    eids_b = np.unique(unit_df_region_b["eid"])

    # here we can filter on significant eids
    eidx = np.intersect1d(eids_a, eids_b)
    if len(eidx) == 1:
        return []
    else:
        return eidx


def run_flattened(list_of_regions, epoch):

    unit_df = bwm_units(one)
    region_pairs = region_combinations(len(list_of_regions))
    all_tasks_to_run = []
    for ra, rb in region_pairs:
        eids = filter_eids_for_region_pair(unit_df, list_of_regions[ra], list_of_regions[rb])
        for eid in tqdm(eids):
            all_tasks_to_run.append((eid, list_of_regions[ra], list_of_regions[rb], epoch))

    print(f"Total tasks: {len(all_tasks_to_run)}")

    processed_results = []
    workers = os.cpu_count() // 4  # type: ignore
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results_iterator = executor.map(prepare_and_run_data, all_tasks_to_run)

        processed_results = list(
            tqdm(results_iterator, total=len(all_tasks_to_run), desc="Processing Tasks")
        )

    print("Now collecting and writing out")

    region_data = {}
    for region, eid, information_pickle in processed_results:

        if region not in region_data:
            region_data[region] = {}

        if information_pickle is not None:
            region_data[region][eid] = information_pickle

    n_bins = config["n_bins_decoding"]
    discretizer = config["discretize_decoding"]

    suffix = ""

    if discretizer == 1:
        suffix += f"_equipopulated_{n_bins}"
    elif discretizer == 2:
        suffix += f"_equispaced_{n_bins}"

    if config["decoder_output_only"] == True:
        suffix += "_outputonly"
    else:
        suffix += "_decomposition"

    # this will make one huge pickle:
    for region, region_pickle in region_data.items():
        with open(f"./data/generated/selective_pairs_{region}_{epoch}_{suffix}.pkl", "wb") as f:
            pkl.dump(region_pickle, f)

    print("Done!")


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

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    config["epoch"] = "stim"
    run_flattened(important_regions, "stim")

    config["epoch"] = "choice"
    run_flattened(important_regions, "choice")
