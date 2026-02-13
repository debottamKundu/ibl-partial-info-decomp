import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score
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
from sklearn.utils import compute_sample_weight
from brainwidemap import bwm_query, bwm_units, load_good_units, load_trials_and_mask
from brainwidemap.bwm_loading import merge_probes
from iblatlas.atlas import AllenAtlas, BrainRegions
from matplotlib import pyplot as plt
from one.api import ONE
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from ibl_info.dual_decoders import (
    run_dual_region_decoder_bootstrapping,
    run_dual_region_decoder_bootstrapping_hyperparamopt,
)
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
    prepare_ephys_data,
    get_new_cinc_intervals_choice,
)
from ibl_info.utils import check_config, equispaced_binning, equipopulated_binning


config = check_config()


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def run_decoder_nested_cv(
    neural_data,
    trial_labels,
    n_splits=5,
    scale=False,
    param_grid=None,
):
    """
    Runs linear decoders using Nested Cross-Validation (No Bootstrapping).

    1. Outer Loop: Splits data into Train/Test (k-fold).
    2. Inner Loop: Optimizes hyperparameters (C) on Train data via GridSearchCV.
    3. Refit: Trains best model on the specific fold's Train data.
    4. Predict: Evaluates on the specific fold's Test data.

    Returns:
        dict: A dictionary containing overall metrics, predictions, and selected parameters.
    """

    # 1. Setup Defaults
    if param_grid is None:
        param_grid = {"clf__C": [0.01, 0.1, 1, 10]}

    X = neural_data
    y = np.array(trial_labels).flatten()
    n_trials, n_neurons = X.shape
    n_classes = len(np.unique(y))

    probs_all = np.zeros((n_trials, n_classes))

    best_params_per_fold = []

    print(f"Starting Nested CV (Outer Splits: {n_splits})...")

    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        train_weights = compute_sample_weight("balanced", y=y_train)

        steps = [("clf", LogisticRegression(solver="liblinear", max_iter=1000))]

        if scale:

            steps.insert(0, ("scaler", StandardScaler()))  # type: ignore

        pipeline = Pipeline(steps)

        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1)

        grid.fit(X_train, y_train, clf__sample_weight=train_weights)

        best_params_per_fold.append(grid.best_params_["clf__C"])

        probs_all[test_idx] = grid.predict_proba(X_test)

    preds_all = np.argmax(probs_all, axis=1)

    results = {
        "predictions": preds_all,
        "targets": y,
        "best_params": best_params_per_fold,
        "accuracy": accuracy_score(y, preds_all),
        "balanced_accuracy": balanced_accuracy_score(y, preds_all),
    }

    return results


def gather_data(session_id, region):

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

    # trials-feedback is the target
    target_variable = trials["feedbackType"]

    # "Quiescent": {
    #     "align": "stimOn_times",
    #     "offset": -0.1,  # Align to -0.1s before Stim
    #     "t_pre": 0.5,
    #     "t_post": 0.0,
    # }

    stimon_times = trials.stimOn_times.values
    qu_time_on = stimon_times - 0.5
    qu_time_off = stimon_times - 0.1
    intervals = np.array([qu_time_on, qu_time_off]).T

    binned_spikes, actual_regions, n_units, cluster_uuids_list = prepare_ephys_data(
        spikes, clusters, intervals, [region], minimum_units=config["min_units_decoding"]
    )  # this returns all neurons from a single region that pass qc

    if len(binned_spikes) == 0:
        print(f'Neurons less than {config["min_units_decoding"]} in {region}')
        return {}

    spike_data = binned_spikes[0]
    target_variable[target_variable == -1] = 0

    return spike_data, target_variable


def run_feedback_decoder(session_id, region, epoch):

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

    # trials-feedback is the target
    target_variable = trials["feedbackType"]

    # "Quiescent": {
    #     "align": "stimOn_times",
    #     "offset": -0.1,  # Align to -0.1s before Stim
    #     "t_pre": 0.5,
    #     "t_post": 0.0,
    # }

    stimon_times = trials.stimOn_times.values
    qu_time_on = stimon_times - 0.5
    qu_time_off = stimon_times - 0.1
    intervals = np.array([qu_time_on, qu_time_off]).T

    binned_spikes, actual_regions, n_units, cluster_uuids_list = prepare_ephys_data(
        spikes, clusters, intervals, [region], minimum_units=config["min_units_decoding"]
    )  # this returns all neurons from a single region that pass qc

    if len(binned_spikes) == 0:
        print(f'Neurons less than {config["min_units_decoding"]} in {region}')
        return {}

    spike_data = binned_spikes[0]
    target_variable[target_variable == -1] = 0

    results = run_decoder_nested_cv(
        spike_data,
        target_variable,
        n_splits=5,
        scale=False,
    )

    null_scores = []
    y_shuffled = np.array(target_variable).copy().flatten()
    n_permutations = config["permutations"]
    print(f"Starting {n_permutations} permutations...")

    for i in range(n_permutations):
        print(f"  Permutation {i + 1}/{n_permutations}...")

        # SHUFFLE: Destroy the relationship between X and y
        np.random.shuffle(y_shuffled)

        # Run decoder on shuffled labels
        # Note: We pass neural_data unchanged, but y is shuffled
        perm_results = run_decoder_nested_cv(spike_data, y_shuffled)

        null_scores.append(perm_results["balanced_accuracy"])

    null_scores = np.array(null_scores)

    result_pickle = {
        "results": results,
        "null_scores": null_scores,
    }
    return result_pickle


def prepare_and_run_data(args):
    eid, region, epoch = args

    try:
        decoding_pickle = run_feedback_decoder(eid, region, epoch)
        if decoding_pickle == {}:
            return region, eid, None
    except Exception as e:
        print(f"Error in {eid}: {e}")
        decoding_pickle = None

    return region, eid, decoding_pickle


def run_flattened(list_of_regions, epoch):

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
    for region, eid, decoding_pickle in processed_results:

        if region not in region_data:
            region_data[region] = {}

        if decoding_pickle is not None:
            region_data[region][eid] = decoding_pickle

    # this will make one huge pickle:
    for region, region_pickle in region_data.items():
        with open(f"./data/generated/selective_{region}_{epoch}_feedback_decoder.pkl", "wb") as f:
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

    run_flattened(important_regions, "quiescent")
