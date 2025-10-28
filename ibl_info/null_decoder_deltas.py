# 1. IMPORTS AND SETUP
# =============================================================================
import logging
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
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


# 2. LOGGING CONFIGURATION
# =============================================================================
def setup_logger():
    """Configures a logger to output to both console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("analysis.log", mode="w"),  # Log to a file
            logging.StreamHandler(),  # Log to the console
        ],
    )
    return logging.getLogger(__name__)


# 3. MODEL DEFINITIONS
# =============================================================================
def define_models_and_parameters():
    """Defines scikit-learn pipelines and hyperparameter grids for each model."""
    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(solver="liblinear", random_state=42, class_weight="balanced"),
            ),
        ]
    )
    lr_params = {"clf__C": np.logspace(-3, 3, 7)}

    rf_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced"),
            ),  # n_jobs=1 inside, parallelize outside
        ]
    )
    rf_params = {
        "clf__n_estimators": [50, 150],
        "clf__max_depth": [3, 5],
        "clf__min_samples_leaf": [3, 5],
        "clf__max_features": ["sqrt"],
    }
    models = {"Logistic Regression": (lr_pipe, lr_params), "Random Forest": (rf_pipe, rf_params)}
    return models


# 4. CORE ANALYSIS FUNCTIONS
# =============================================================================
def get_performance_and_null_distribution(X, y, pipeline, param_grid, n_permutations=20):
    """Core analysis engine: nested CV + permutation testing for one dataset."""
    # Note: GridSearchCV can be parallelized, but we parallelize at the mouse level for bigger gains.
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Real Performance Score ---
    real_scores = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        grid_search = GridSearchCV(
            estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=1
        )
        grid_search.fit(X_train, y_train)
        score = roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1])
        real_scores.append(score)
    real_performance = np.mean(real_scores)

    # --- Null Distribution ---
    null_scores = []
    for i in range(n_permutations):
        y_permuted = np.random.permutation(y)
        permuted_run_scores = []
        for train_idx, test_idx in outer_cv.split(X, y_permuted):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train_p, y_test_p = y_permuted[train_idx], y_permuted[test_idx]
            gs_null = GridSearchCV(
                estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=1
            )
            gs_null.fit(X_train, y_train_p)
            score_null = roc_auc_score(y_test_p, gs_null.predict_proba(X_test)[:, 1])
            permuted_run_scores.append(score_null)
        null_scores.append(np.mean(permuted_run_scores))
    return real_performance, np.array(null_scores)


def get_effect_size_for_condition(X, y, model_name, models_to_run, n_perms):
    """Wrapper to run the core analysis and return a single effect size."""
    pipeline, params = models_to_run[model_name]
    real_score, null_dist = get_performance_and_null_distribution(
        X,
        y,
        pipeline,
        params,
        n_perms,
    )
    mean_null, std_null = np.mean(null_dist), np.std(null_dist)
    effect_size = (real_score - mean_null) / std_null if std_null > 0 else 0.0
    return effect_size


def stratified_subsample_generator(congruent_X, congruent_y, incongruent_y, n_repeats=3):
    """Generator for creating stratified subsamples of a congruentity dataset."""
    target_size = len(incongruent_y)
    left_fraction = np.sum(incongruent_y == 1) / target_size
    congruent_left_idx = np.where(congruent_y == 1)[0]
    congruent_right_idx = np.where(congruent_y == 0)[0]

    for i in range(n_repeats):
        n_left = int(np.round(left_fraction * target_size))
        n_right = target_size - n_left
        selected_left = np.random.choice(congruent_left_idx, n_left, replace=False)
        selected_right = np.random.choice(congruent_right_idx, n_right, replace=False)
        selected_indices = np.concatenate((selected_left, selected_right))
        np.random.shuffle(selected_indices)
        yield congruent_X[selected_indices, :], congruent_y[selected_indices]


# 5. SINGLE MOUSE ANALYSIS FUNCTION
# =============================================================================
def analyze_single_mouse(session_id, epoch, one, region, config):
    """
    Orchestrates the entire analysis for one mouse and returns a results dictionary.
    """
    logger = logging.getLogger(__name__)

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
        spikes, clusters, intervals, [region], minimum_units=5
    )  # this returns all neurons from a single region that pass qc
    spike_data = binned_spikes[0].T
    congruent_spikes = spike_data[:, congruent_flags].T
    congruent_targets = target_variable[congruent_flags]
    # incongruent trials
    incongruent_spikes = spike_data[:, incongruent_flags].T
    incongruent_targets = target_variable[incongruent_flags]
    logger.info(f"--- Starting analysis for {session_id} ---")

    # Unpack data and config
    n_perms = config["n_permutations"]
    n_subsamples = config["n_subsamples"]

    # A. Analyze the incongruent once
    logger.info(f"[{session_id}] Analyzing incongruent ({len(incongruent_targets)} trials)...")
    d_LR_incongruent = get_effect_size_for_condition(
        incongruent_spikes, incongruent_targets, "Logistic Regression", models_to_run, n_perms
    )
    d_RF_incongruent = get_effect_size_for_condition(
        incongruent_spikes, incongruent_targets, "Random Forest", models_to_run, n_perms
    )
    delta_d_incongruent = d_RF_incongruent - d_LR_incongruent

    # B. Iteratively analyze subsamples of the larger category
    logger.info(f"[{session_id}] Subsampling  congruent ")

    d_LR_congruent_iters, d_RF_congruent_iters = [], []
    subsampler = stratified_subsample_generator(
        congruent_spikes, congruent_targets, incongruent_targets, n_subsamples
    )

    for i, (X_sub, y_sub) in enumerate(subsampler):
        d_lr = get_effect_size_for_condition(
            X_sub,
            y_sub,
            "Logistic Regression",
            models_to_run,
            n_perms,
        )
        d_rf = get_effect_size_for_condition(
            X_sub,
            y_sub,
            "Random Forest",
            models_to_run,
            n_perms,
        )
        d_LR_congruent_iters.append(d_lr)
        d_RF_congruent_iters.append(d_rf)

    # C. Average results for the larger category
    d_LR_congruent_stable = np.mean(d_LR_congruent_iters)
    d_RF_congruent_stable = np.mean(d_RF_congruent_iters)
    delta_d_congruent = d_RF_congruent_stable - d_LR_congruent_stable

    # D. Calculate final metric
    # Ensure subtraction is (Category A - Category B) regardless of which was smaller

    final_metric = delta_d_incongruent - delta_d_congruent

    logger.info(f"--- Finished analysis for {session_id} ---")

    # Store all results in a dictionary
    return {
        "session_id": session_id,
        "d_LR_incongruent": d_LR_incongruent,
        "d_RF_incongruent": d_RF_incongruent,
        "d_LR_congruent": d_LR_congruent_stable,
        "d_RF_congruent": d_RF_congruent_stable,
        "nonlin_advantage_incongruent": (delta_d_incongruent),
        "nonlin_advantage_congruent": (delta_d_congruent),
        "final_metric": final_metric,
    }


def filter_eids(unit_df, region):
    unit_df_region = unit_df[unit_df["Beryl"] == region]
    eids = np.unique(unit_df_region["eid"])

    return eids


# 6. PARALLELIZATION FUNCTION
# =============================================================================
def run_analysis_in_parallel(all_mice_data, models_to_run, config):
    """Uses joblib to parallelize the single-mouse analysis."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting parallel analysis for {len(all_mice_data)} mice...")

    results = Parallel(n_jobs=-1)(  # Use all available CPU cores
        delayed(analyze_single_mouse)(mouse_data, models_to_run, config)
        for mouse_data in all_mice_data
    )

    logger.info("Parallel analysis complete.")
    return results


def prepare_and_run_data(task_tuple):

    eid, region, epoch, config = task_tuple
    one = ONE()
    try:
        # ideally information pickle, but i want to subsample mutliple times
        information_pickle = analyze_single_mouse(eid, epoch, one, region, config)
        if information_pickle == {}:
            return region, eid, None
        else:
            return region, eid, information_pickle
    except Exception as e:
        print(f"Error regarding {eid} in region {region}: {e}")
        return region, eid, None


def run_flattened(list_of_regions, epoch, config):
    one = ONE()
    unit_df = bwm_units(one)
    all_tasks_to_run = []
    for region in list_of_regions:
        selective_eids = filter_eids(unit_df, region)
        for eid in tqdm(selective_eids):
            all_tasks_to_run.append((eid, region, epoch, config))

    # run parallel here i think
    print(f"Total tasks: {len(all_tasks_to_run)}")

    processed_results = []
    workers = os.cpu_count() // 4  # type: ignore
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results_iterator = executor.map(prepare_and_run_data, all_tasks_to_run)

        processed_results = list(
            tqdm(results_iterator, total=len(all_tasks_to_run), desc="Processing Tasks")
        )

    region_data = {}
    for region, eid, information_pickle in processed_results:

        if region not in region_data:
            region_data[region] = {}

        if information_pickle is not None:
            region_data[region][eid] = information_pickle

    for region, region_pickle in region_data.items():
        with open(f"./data/generated/decoder_null_delta_{region}_.pkl", "wb") as f:
            pkl.dump(region_pickle, f)

    print("Done!")


# 7. MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == "__main__":
    logger = setup_logger()

    # --- Analysis Configuration ---

    config = {
        "n_permutations": 10,  # Use 500-1000 for publication
        "n_subsamples": 3,  # Use 50-100 for publication
    }

    # --- Define Models ---
    models_to_run = define_models_and_parameters()

    important_regions = ["VISp", "IRN"]

    # save this for each region

    # --- Run Full Analysis ---
    # TODO: fix this

    # You can also save your results to a file for later plotting/analysis
    # import pandas as pd
    # results_df = pd.DataFrame(all_results)
    # results_df.to_csv("analysis_results.csv", index=False)
    # logger.info("Detailed results saved to analysis_results.csv")
