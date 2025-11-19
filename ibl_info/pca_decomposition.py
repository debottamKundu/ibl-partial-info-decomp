import itertools
import os
import pickle as pkl
from pathlib import Path
import concurrent.futures
import functools
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
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from ibl_info.selective_decomposition import filter_eids
from sklearn.svm import SVC
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
import ibl_info.measures.information_measures as info
from ibl_info.prepare_data_pid import (
    get_new_cinc_intervals,
    get_new_cinc_intervals_choice,
    prepare_ephys_data,
)
from ibl_info.utils import check_config
from ibl_info.pcaprojections import analyze_neural_interaction

config = check_config()


def run_pca_single_session(session_id, epoch, one, region):

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
    minimum_units = config["min_units"]

    binned_spikes, actual_regions, n_units, cluster_uuids_list = prepare_ephys_data(
        spikes, clusters, intervals, [region], minimum_units=minimum_units
    )  # this returns all neurons from a single region that pass qc

    # no discretization, use raw counts
    spike_data = binned_spikes[0].T
    congruent_target = target_variable[congruent_flags]
    incongruent_target = target_variable[incongruent_flags]

    congruent_spikes = spike_data[:, congruent_flags]
    incongruent_spikes = spike_data[:, incongruent_flags]

    information_pickle["neurons"] = spike_data.shape[0]
    information_pickle["trials"] = trial_count

    information_pickle["all"] = analyze_neural_interaction(
        spike_data, target_variable
    )  # expects neurons x trials
    information_pickle["congruent"] = analyze_neural_interaction(
        congruent_spikes, congruent_target
    )
    information_pickle["incongruent"] = analyze_neural_interaction(
        incongruent_spikes, incongruent_target
    )

    return information_pickle


def prepare_and_run_data(task_tuple):

    eid, region, epoch = task_tuple
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
    )
    try:
        # ideally information pickle, but i want to subsample mutliple times
        information_pickle = run_pca_single_session(
            eid,
            epoch,
            one,
            region,
        )
        if information_pickle == {}:
            return region, eid, None
        else:
            return region, eid, information_pickle
    except Exception as e:
        print(f"Error regarding {eid} in region {region}: {e}")
        return region, eid, None


def run_flattened(list_of_regions, epoch):
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
    )
    unit_df = bwm_units(one)
    all_tasks_to_run = []
    for region in list_of_regions:
        selective_eids = filter_eids(unit_df, region, significant_filter=config["decoder_filter"])
        for eid in tqdm(selective_eids):
            all_tasks_to_run.append((eid, region, epoch))

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

    suffix = "lineardecoder_goodsessions"

    for region, region_pickle in region_data.items():
        with open(
            f"./data/generated/selective_decomposition_{region}_{epoch}_{suffix}.pkl", "wb"
        ) as f:
            pkl.dump(region_pickle, f)

    print("Done!")


if __name__ == "__main__":

    important_regions = config["stim_prior_regions"]
    run_flattened(important_regions, "stim")
