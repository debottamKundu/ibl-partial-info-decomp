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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier
from ibl_info.selective_decomposition import filter_eids
from sklearn.svm import SVC
from tqdm import tqdm

import ibl_info.measures.information_measures as info
from ibl_info.prepare_data_pid import get_new_cinc_intervals, prepare_ephys_data
from ibl_info.utils import check_config, equipopulated_binning, equispaced_binning

config = check_config()


def split_and_decode(
    trial_types,
    neural_activity,
    n_splits=10,
    decoder="logreg",
    return_probs=True,
    random_state=None,
):
    """
    Perform multiple random neuron splits and run LOOCV decoders for each half.

    Parameters
    ----------
    trial_types : array-like, shape (n_trials,)
        Labels for each trial.
    neural_activity : array-like, shape (n_trials, n_neurons)
        Neural activity per trial.
    n_splits : int, default=10
        Number of random neuron splits to perform.
    decoder : {"logreg", "svm"}, default="logreg"
        Choice of decoder model: logistic regression or SVM.
    return_probs : bool, default=True
        If True, return class probabilities; if False, return hard predictions.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    outputs_split1 : ndarray, shape (n_splits, n_trials, n_classes or 1)
    outputs_split2 : ndarray, shape (n_splits, n_trials, n_classes or 1)
    """
    rng = np.random.default_rng(random_state)
    trial_types = np.asarray(trial_types)
    X = np.asarray(neural_activity)
    n_trials, n_neurons = X.shape
    n_classes = len(np.unique(trial_types))

    # Storage for outputs
    if return_probs:
        outputs1 = np.zeros((n_splits, n_trials, n_classes))
        outputs2 = np.zeros((n_splits, n_trials, n_classes))
    else:
        outputs1 = np.zeros((n_splits, n_trials), dtype=int)
        outputs2 = np.zeros((n_splits, n_trials), dtype=int)

    accuracies = np.zeros((n_splits, 2))
    loo = LeaveOneOut()

    for s in range(n_splits):
        # Random neuron split
        perm = rng.permutation(n_neurons)
        half = n_neurons // 2
        idx1, idx2 = perm[:half], perm[half:]
        X1, X2 = X[:, idx1], X[:, idx2]

        # Storage per split
        if return_probs:
            out1 = np.zeros((n_trials, n_classes))
            out2 = np.zeros((n_trials, n_classes))
        else:
            out1 = np.zeros(n_trials, dtype=int)
            out2 = np.zeros(n_trials, dtype=int)

        for train_idx, test_idx in tqdm(loo.split(X1)):
            # Pick decoder
            if decoder == "logreg":
                clf1 = LogisticRegression(max_iter=500, solver="lbfgs")
                clf2 = LogisticRegression(max_iter=500, solver="lbfgs")
            elif decoder == "svm":
                clf1 = SVC(kernel="linear", probability=return_probs)
                clf2 = SVC(kernel="linear", probability=return_probs)
            elif decoder == "nonlinear":
                clf1 = SVC(kernel="rbf", probability=return_probs)
                clf2 = SVC(kernel="rbf", probability=return_probs)
            else:
                raise ValueError("decoder must be 'logreg' or 'svm'")

            clf1.fit(X1[train_idx], trial_types[train_idx])
            clf2.fit(X2[train_idx], trial_types[train_idx])

            if return_probs:
                out1[test_idx] = clf1.predict_proba(X1[test_idx])
                out2[test_idx] = clf2.predict_proba(X2[test_idx])
            else:
                out1[test_idx] = clf1.predict(X1[test_idx])
                out2[test_idx] = clf2.predict(X2[test_idx])

            if return_probs:
                # Convert probabilities to class labels (using argmax)
                preds1 = np.argmax(out1, axis=1)
                preds2 = np.argmax(out2, axis=1)
            else:
                preds1 = out1
                preds2 = out2
        accuracy1 = balanced_accuracy_score(trial_types, preds1)
        accuracy2 = balanced_accuracy_score(trial_types, preds2)

        accuracies[s] = np.asarray([accuracy1, accuracy2])  # type: ignore

        outputs1[s] = out1
        outputs2[s] = out2

    return outputs1, outputs2, accuracies


def compute_decoder_pid(target, spikes, n_bins=2):

    output_a, output_b, accuracies = split_and_decode(target, spikes)
    n_bins = config["n_bins"]
    # output_a is splits x trials x 2
    # we only take probability left ( I think?)
    # probability left is index 1

    probability_output_a = output_a[:, :, 1]
    probability_output_b = output_b[:, :, 1]

    repeats = output_a.shape[0]

    pid_array = np.zeros((repeats, 6))

    for idx in range(0, repeats):
        X1 = np.asarray(
            equispaced_binning(probability_output_a[idx], n_bins=n_bins), dtype=np.int32
        )
        X2 = np.asarray(
            equispaced_binning(probability_output_b[idx], n_bins=n_bins), dtype=np.int32
        )

        Y = np.asarray(target, dtype=np.int32)

        pid_array[idx, 0:4] = info.corrected_pid(sourcea=X1, sourceb=X2, target=Y)  # type: ignore # QE
        pid_array[idx, 4:] = accuracies[idx]

    return pid_array


def subsampled(congruent_spikes, congruent_targets, incongruent_targets, decoder_pid=True):

    left_fraction = np.sum(incongruent_targets == 1) / len(incongruent_targets)

    # we want to ensure similar fraction for congruent subsampling
    left_congruent = np.where(congruent_targets == 1)[0]
    right_congruent = np.where(congruent_targets == 0)[0]

    sampled_pid = []
    for repeats in range(config["repeats_for_bias_correction"]):
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
        if decoder_pid:
            pid_array = compute_decoder_pid(subsampled_targets, subsampled_spikes.T)
            sampled_pid.append(pid_array)
        else:
            linear, nonlinear, performance_delta = linear_nonlinear_delta(
                subsampled_targets, subsampled_spikes.T
            )
            sampled_pid.append([linear, nonlinear, performance_delta])

    # average
    sampled_pid = np.asarray(sampled_pid)

    # sampled_pid = np.mean(sampled_pid, axis=0)

    return sampled_pid


def linear_nonlinear_delta(
    trial_types,
    neural_activity,
    linear_model="logreg",
    nonlinear_model="random_forest",
    metric="accuracy",
):
    """
    Compare performance of linear vs nonlinear decoders with LOOCV.

    Parameters
    ----------
    trial_types : array-like, shape (n_trials,)
        Labels for each trial.
    neural_activity : array-like, shape (n_trials, n_neurons)
        Neural activity per trial.
    linear_model : {"logreg", "svm_linear"}, default="logreg"
        Choice of linear classifier.
    nonlinear_model : {"svm_rbf", "mlp", "random_forest"}, default="svm_rbf"
        Choice of nonlinear classifier.
    metric : {"accuracy"}, default="accuracy"
        Performance metric to compare.

    Returns
    -------
    perf_linear : float
        Performance of linear model.
    perf_nonlinear : float
        Performance of nonlinear model.
    diff : float
        Difference (nonlinear - linear).
    """
    trial_types = np.asarray(trial_types)
    X = np.asarray(neural_activity)
    n_trials = X.shape[0]

    loo = LeaveOneOut()
    preds_linear, preds_nonlin = [], []

    for train_idx, test_idx in loo.split(X):
        # ----- Linear model -----
        if linear_model == "logreg":
            clf_lin = LogisticRegression(max_iter=1000, solver="lbfgs")
        elif linear_model == "svm_linear":
            clf_lin = SVC(kernel="linear")
        else:
            raise ValueError("linear_model must be 'logreg' or 'svm_linear'")

        clf_lin.fit(X[train_idx], trial_types[train_idx])
        preds_linear.append(clf_lin.predict(X[test_idx])[0])

        # ----- Nonlinear model -----
        if nonlinear_model == "svm_rbf":
            clf_nonlin = SVC(kernel="rbf")
        elif nonlinear_model == "mlp":
            clf_nonlin = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
        elif nonlinear_model == "random_forest":
            clf_nonlin = RandomForestClassifier()
        else:
            raise ValueError("nonlinear_model must be 'svm_rbf','random_forest' or 'mlp'")

        clf_nonlin.fit(X[train_idx], trial_types[train_idx])
        preds_nonlin.append(clf_nonlin.predict(X[test_idx])[0])

    # Compute performance
    if metric == "accuracy":
        perf_lin = balanced_accuracy_score(trial_types, preds_linear)
        perf_nonlin = balanced_accuracy_score(trial_types, preds_nonlin)
    else:
        raise ValueError("Only accuracy metric implemented right now.")

    return perf_lin, perf_nonlin, perf_nonlin - perf_lin


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

    intervals, target_variable, congruent_flags, incongruent_flags = get_new_cinc_intervals(
        trials, epoch
    )

    binned_spikes, actual_regions, n_units, cluster_uuids_list = prepare_ephys_data(
        spikes, clusters, intervals, [region], minimum_units=5
    )  # this returns all neurons from a single region that pass qc
    # however, it is in trials x neurons

    spike_data = binned_spikes[0].T

    congruent_target = target_variable[congruent_flags]
    incongruent_target = target_variable[incongruent_flags]

    congruent_spikes = spike_data[:, congruent_flags]
    incongruent_spikes = spike_data[:, incongruent_flags]

    trial_count = np.zeros((3))

    trial_count[0] = intervals.shape[0]
    trial_count[1] = np.sum(congruent_flags)
    trial_count[2] = np.sum(incongruent_flags)

    # now we add the new decoder functions
    information_pickle = {}
    information_pickle["neurons"] = spike_data.shape[0]
    information_pickle["trials"] = trial_count

    # information_pickle["incongruent_pid"] = compute_decoder_pid(
    #     incongruent_target, incongruent_spikes.T
    # )

    # information_pickle["congruent_pid"] = subsampled(
    #     congruent_spikes, congruent_target, incongruent_target
    # )

    information_pickle["incongruent_delta"] = linear_nonlinear_delta(
        incongruent_target, incongruent_spikes.T
    )

    information_pickle["congruent_delta"] = subsampled(
        congruent_spikes, congruent_target, incongruent_target, decoder_pid=False
    )

    return information_pickle


def prepare_and_run_data(task_tuple):

    eid, region, epoch = task_tuple
    one = ONE()
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

    one = ONE()
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

    suffix = "decoder_good_decoder_sessions_only"

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
