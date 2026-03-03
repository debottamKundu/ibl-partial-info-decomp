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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

config = check_config()


def run_nested_decoder(X, y, cv_splits, n_permutations=50, random_state=42):
    """
    Runs a 2-step (nested) cross-validation logistic regression decoder.

    Parameters:
    -----------
    X : array-like of shape (n_trials, n_neurons)
        The neural spiking data.
    y : array-like of shape (n_trials,)
        The feedback labels (-1, 1).
    cv_splits : list of tuples or scikit-learn CV generator
        The outer cross-validation splits. e.g., list(StratifiedKFold(5).split(X, y))
    n_permutations : int
        Number of shuffles for the null distribution.

    Returns:
    --------
    results : dict
        Contains 'fold_scores', 'mean_score', 'null_distribution', and 'p_value'.
    """
    cv_splits = list(cv_splits)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    param_grid = {
        "logreg__penalty": ["l1", "l2"],
        "logreg__C": [0.001, 0.01, 0.1, 1.0, 10.0],
    }

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    clf_tuned = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    print("Evaluating true model across outer folds...")
    fold_scores = cross_val_score(
        estimator=clf_tuned,
        X=X,
        y=y,
        cv=cv_splits,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    mean_score = np.mean(fold_scores)
    print(f"True Mean Balanced Accuracy: {mean_score:.3f}")

    print(
        f"Running permutation test with {n_permutations} shuffles (this may take a few minutes)..."
    )

    _, null_distribution, p_value = permutation_test_score(
        estimator=clf_tuned,
        X=X,
        y=y,
        cv=cv_splits,
        scoring="balanced_accuracy",
        n_permutations=n_permutations,
        n_jobs=-1,
        random_state=random_state,
    )
    print(f"Mean Null Balanced Accuracy: {np.mean(null_distribution):.3f} (p = {p_value:.4f})")

    return {
        "fold_scores": fold_scores,
        "mean_score": mean_score,
        "null_distribution": null_distribution,
        "p_value": p_value,
    }


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
    # target_variable[target_variable == -1] = 0

    results = run_nested_decoder(spike_data, target_variable, cv_splits=3)

    return results


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
        with open(
            f"./data/generated/selective_{region}_{epoch}_feedback_decoder_new.pkl", "wb"
        ) as f:
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
