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
from ibl_info.utils import check_config, equipopulated_binning, equispaced_binning


config = check_config()


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def run_decoder_bootstrapping(
    neural_data,
    trial_labels,
    subset_size_D,
    n_bootstraps=50,
    n_splits=5,
    congruent_mask=None,
    incongruent_mask=None,
    scale=False,
):
    """
    Bootstraps linear decoders on non-overlapping subsets of neurons.
    Uses K-Fold Cross Validation to ensure predictions are generated for every trial.
    Includes data scaling (StandardScaler) within the CV loop.

    Parameters:
    -----------
    neural_data : np.ndarray
        Shape (n, m) where n is neurons and m is trials.
    trial_labels : np.ndarray
        Shape (1, n) or (n,). The class labels for the trials.
    subset_size_D : int
        Number of neurons to include in each of the two non-overlapping sets.
    n_bootstraps : int
        Number of times to resample neurons.
    n_splits : int
        Number of K-Fold splits.
    congruent_mask : np.ndarray, optional
        Boolean mask indicating congruent trials.
    incongruent_mask : np.ndarray, optional
        Boolean mask indicating incongruent trials.
    scale: bool, optional
        Set if scaling is on or not

    Returns:
    --------
    list of dicts
        Each dict contains:
        - 'probs_A': Full set of probabilities for Set A (aligned with input trials)
        - 'probs_B': Full set of probabilities for Set B (aligned with input trials)
        - 'probs_A_cong': Probabilities for Set A, subsetted to congruent trials
        - 'probs_A_incong': Probabilities for Set A, subsetted to incongruent trials
        - 'probs_B_cong': Probabilities for Set B, subsetted to congruent trials
        - 'probs_B_incong': Probabilities for Set B, subsetted to incongruent trials
        - 'y_cong': Ground truth labels for congruent trials
        - 'y_incong': Ground truth labels for incongruent trials
        - 'accuracy_A': Overall accuracy of Decoder A
        - 'accuracy_B': Overall accuracy of Decoder B
        - 'balanced_acc_A': Balanced accuracy of Decoder A
        - 'balanced_acc_B': Balanced accuracy of Decoder B
        - 'neurons_A_indices': Indices of neurons used for Set A
        - 'neurons_B_indices': Indices of neurons used for Set B
    """

    X = neural_data  # (n_trials, n_neurons)
    y = trial_labels.flatten()

    n_trials, n_neurons = X.shape

    # Handle masks: Ensure they are flattened boolean arrays
    cong_indices = None
    incong_indices = None

    if congruent_mask is not None:
        congruent_mask = np.array(congruent_mask).flatten().astype(bool)
        cong_indices = np.where(congruent_mask)[0]

    if incongruent_mask is not None:
        incongruent_mask = np.array(incongruent_mask).flatten().astype(bool)
        incong_indices = np.where(incongruent_mask)[0]

    # Check constraints
    if 2 * subset_size_D > n_neurons:
        raise ValueError(
            f"Cannot select 2 non-overlapping sets of size {subset_size_D} "
            f"from {n_neurons} neurons."
        )

    results = []

    print(f"Starting bootstrapping: {n_bootstraps} iterations with {n_splits}-Fold CV...")

    for i in range(n_bootstraps):
        # 2. Sub-sample neurons
        permuted_indices = np.random.permutation(n_neurons)
        idx_A = permuted_indices[:subset_size_D]
        idx_B = permuted_indices[subset_size_D : 2 * subset_size_D]

        X_subset_A = X[:, idx_A]
        X_subset_B = X[:, idx_B]

        # Placeholders for full dataset predictions
        n_classes = len(np.unique(y))

        # Initialize with zeros.
        # Since we iterate through all folds, every index will be filled exactly once.
        probs_A_all = np.zeros((n_trials, n_classes))
        probs_B_all = np.zeros((n_trials, n_classes))

        # 3. K-Fold Cross Validation Loop
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

        for train_idx, test_idx in skf.split(X, y):
            # Split Data
            X_train_A, X_test_A = X_subset_A[train_idx], X_subset_A[test_idx]
            X_train_B, X_test_B = X_subset_B[train_idx], X_subset_B[test_idx]
            y_train = y[train_idx]

            if scale:
                scaler_A = StandardScaler()
                X_train_A = scaler_A.fit_transform(X_train_A)
                X_test_A = scaler_A.transform(X_test_A)

                scaler_B = StandardScaler()
                X_train_B = scaler_B.fit_transform(X_train_B)
                X_test_B = scaler_B.transform(X_test_B)

            # --- SAMPLE WEIGHTS ---
            train_weights = compute_sample_weight(class_weight="balanced", y=y_train)

            # Train Decoders
            clf_A = LogisticRegression(solver="lbfgs", max_iter=1000)
            clf_B = LogisticRegression(solver="lbfgs", max_iter=1000)

            clf_A.fit(X_train_A, y_train, sample_weight=train_weights)
            clf_B.fit(X_train_B, y_train, sample_weight=train_weights)

            # Predict and Fill
            # test_idx here corresponds to the indices in the ORIGINAL X array.
            # So probs_A_all will remain aligned with the original trial order.
            probs_A_all[test_idx] = clf_A.predict_proba(X_test_A)
            probs_B_all[test_idx] = clf_B.predict_proba(X_test_B)

        # 4. Calculate Metrics (Overall)
        preds_A = np.argmax(probs_A_all, axis=1)
        preds_B = np.argmax(probs_B_all, axis=1)

        acc_A = accuracy_score(y, preds_A)
        acc_B = accuracy_score(y, preds_B)
        bal_acc_A = balanced_accuracy_score(y, preds_A)
        bal_acc_B = balanced_accuracy_score(y, preds_B)

        metrics_sub = {}

        # Helper to compute subset metrics safely
        def get_subset_metrics(indices, y_full, preds_A_full, preds_B_full, prefix):
            if indices is None or len(indices) == 0:
                return {}

            y_sub = y_full[indices]
            p_A_sub = preds_A_full[indices]
            p_B_sub = preds_B_full[indices]

            return {
                f"accuracy_A_{prefix}": accuracy_score(y_sub, p_A_sub),
                f"accuracy_B_{prefix}": accuracy_score(y_sub, p_B_sub),
                f"balanced_acc_A_{prefix}": balanced_accuracy_score(y_sub, p_A_sub),
                f"balanced_acc_B_{prefix}": balanced_accuracy_score(y_sub, p_B_sub),
            }

        metrics_sub.update(get_subset_metrics(cong_indices, y, preds_A, preds_B, "cong"))
        metrics_sub.update(get_subset_metrics(incong_indices, y, preds_A, preds_B, "incong"))

        # 5. Store Results
        # Explicitly slicing the results here so you don't have to worry about indices later.
        run_data = {
            "iteration": i,
            # Full aligned arrays
            "probs_A": probs_A_all,
            "probs_B": probs_B_all,
            # Subsetted arrays (convenience)
            "probs_A_cong": probs_A_all[cong_indices] if cong_indices is not None else None,
            "probs_A_incong": probs_A_all[incong_indices] if incong_indices is not None else None,
            "probs_B_cong": probs_B_all[cong_indices] if cong_indices is not None else None,
            "probs_B_incong": probs_B_all[incong_indices] if incong_indices is not None else None,
            # Metrics
            "accuracy_A": acc_A,
            "accuracy_B": acc_B,
            "balanced_acc_A": bal_acc_A,
            "balanced_acc_B": bal_acc_B,
            # Metadata
            "y_true": y,
            "y_cong": y[cong_indices] if cong_indices is not None else None,
            "y_incong": y[incong_indices] if incong_indices is not None else None,
            "neurons_A_indices": idx_A,
            "neurons_B_indices": idx_B,
        }
        run_data.update(metrics_sub)

        results.append(run_data)

    print("Bootstrapping complete.")
    return results


def compute_information_metrics(target, sourcea, sourceb):
    mi_a = info.corrected_mutual_information(source=sourcea, target=target)
    mi_b = info.corrected_mutual_information(source=sourceb, target=target)
    tvmi_ab = info.correct_trivariate_mi(source_a=sourcea, source_b=sourceb, target=target)
    pid_ab = info.corrected_pid(sourcea=sourcea, sourceb=sourceb, target=target)

    return np.concatenate([[mi_a, mi_b, tvmi_ab], pid_ab])  # type: ignore # 7 elements


def compute_decoder_pid(
    target, spikes, n_bootstaps=50, n_bins=5, congruent_mask=None, incongruent_mask=None
):

    results = run_decoder_bootstrapping(
        neural_data=spikes,
        trial_labels=target,
        subset_size_D=10,
        n_bootstraps=n_bootstaps,
        n_splits=5,
        congruent_mask=congruent_mask,
        incongruent_mask=incongruent_mask,
    )
    # save results (yes), return this

    # skip the all trials
    information_array = np.zeros((n_bootstaps, 2, 7))

    for iteration in range(n_bootstaps):

        # NOTE: use this sometime

        output_a_all = results[iteration]["probs_A"]
        output_b_all = results[iteration]["probs_B"]
        target_all = results[iteration]["y_true"]

        output_a_con = results[iteration]["probs_A_cong"]
        output_b_con = results[iteration]["probs_B_cong"]
        target_con = results[iteration]["y_cong"]

        output_a_incon = results[iteration]["probs_A_incong"]
        output_b_incon = results[iteration]["probs_B_incong"]
        target_incon = results[iteration]["y_incong"]

        X1_con = np.asarray(equispaced_binning(output_a_con[:, 0], n_bins=n_bins), dtype=np.int32)
        X2_con = np.asarray(equispaced_binning(output_b_con[:, 0], n_bins=n_bins), dtype=np.int32)

        Y_con = np.asarray(target_con, dtype=np.int32)

        X1_incon = np.asarray(
            equispaced_binning(output_a_incon[:, 0], n_bins=n_bins), dtype=np.int32
        )
        X2_incon = np.asarray(
            equispaced_binning(output_b_incon[:, 0], n_bins=n_bins), dtype=np.int32
        )
        Y_incon = np.asarray(target_incon, dtype=np.int32)

        information_array[iteration, 0, :] = compute_information_metrics(  # type: ignore
            Y_con, X1_con, X2_con
        )

        information_array[iteration, 1, :] = compute_information_metrics(  # type: ignore
            Y_incon, X1_incon, X2_incon
        )

    return information_array, results


def run_decoder_single_session(session_id, epoch, one, region):

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

    binned_spikes, actual_regions, n_units, cluster_uuids_list = prepare_ephys_data(
        spikes, clusters, intervals, [region], minimum_units=config["min_units_decoding"]
    )  # this returns all neurons from a single region that pass qc

    if len(binned_spikes) == 0:
        print(f'Neurons less than {config["min_units_decoding"]} in {region}')
        return {}

    spike_data = binned_spikes[0]  # trials x neurons, what we need
    trial_count = np.zeros((3))

    trial_count[0] = intervals.shape[0]
    trial_count[1] = np.sum(congruent_flags)
    trial_count[2] = np.sum(incongruent_flags)

    # now we add the new decoder functions
    information_pickle = {}
    information_pickle["neurons"] = spike_data.shape[0]
    information_pickle["trials"] = trial_count

    information_results, results = compute_decoder_pid(
        target=target_variable,
        spikes=spike_data,
        n_bootstaps=config["n_bootstraps_decoding"],
        n_bins=config["n_bins_decoding"],
        congruent_mask=congruent_flags,
        incongruent_mask=incongruent_flags,
    )

    information_pickle["information"] = information_results
    information_pickle["decoding_results"] = results

    return information_pickle


def prepare_and_run_data(task_tuple):

    eid, region, epoch, one = task_tuple
    # one = ONE(
    #     base_url="https://openalyx.internationalbrainlab.org",
    #     username="intbrainlab",
    #     password="international",
    # )
    try:
        # ideally information pickle, but i want to subsample mutliple times
        information_pickle = run_decoder_single_session(
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
        password="international",
        silent=True,
        username="intbrainlab",
    )
    unit_df = bwm_units(one)
    all_tasks_to_run = []
    for region in list_of_regions:
        selective_eids = filter_eids(unit_df, region, significant_filter=config["decoder_filter"])
        for eid in tqdm(selective_eids):
            all_tasks_to_run.append((eid, region, epoch, one))

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

    suffix = "decoder_alldata_goodsessions_projections"

    # this will make one huge pickle:
    for region, region_pickle in region_data.items():
        with open(
            f"./data/generated/selective_decomposition_{region}_{epoch}_{suffix}.pkl", "wb"
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

    run_flattened(important_regions, "stim")
    # one = ONE()
    # unit_df = bwm_units(one)
    # region = "ZI"
    # selective_eids = filter_eids(unit_df, region)

    # session_id = selective_eids[0]
    # ipickle = run_decoder_single_session(
    #     session_id=session_id, epoch="stim", one=one, region=region
    # )
