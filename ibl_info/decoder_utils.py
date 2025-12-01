# ESTIMATOR_KWARGS = {
#     "tol": 0.0001,
#     "max_iter": 20000,
#     "fit_intercept": True,
# }  # default args for decoder
# HPARAM_GRID = {
#     "alpha": np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
# }  # hyperparameter values to search over
# ESTIMATOR_KWARGS = {
#     "tol": 0.0001,
#     "max_iter": 20000,
#     "fit_intercept": True,
# }  # default args for decoder
from glob import glob
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
from ibl_info.utils import (
    check_config,
    equipopulated_binning,
    equipopulated_binning,
    equispaced_binning,
)
from ibl_info.decoder_pid import compute_information_metrics
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

config = check_config()

# code to load pickles and redo decoding and save.


def load_pickle(file):

    with open(file, "rb") as f:
        return pkl.load(f)


def recompute(data, n_bins=3):
    recomputed_data = {}
    for animal_id in tqdm(data.keys(), desc="Animals"):
        animal = data[animal_id]
        results = animal["decoding_results"]
        n_bootstraps = len(results)

        information_array = np.zeros((n_bootstraps, 2, 7))
        for iteration in tqdm(range(n_bootstraps), desc="Bootstraps", leave=False):
            temp = {}

            output_a_all = results[iteration]["probs_A"]
            output_b_all = results[iteration]["probs_B"]
            target_all = results[iteration]["y_true"]

            output_a_con = results[iteration]["probs_A_cong"]
            output_b_con = results[iteration]["probs_B_cong"]
            target_con = results[iteration]["y_cong"]

            output_a_incon = results[iteration]["probs_A_incong"]
            output_b_incon = results[iteration]["probs_B_incong"]
            target_incon = results[iteration]["y_incong"]

            discretization_type = config["discretize_decoding"]

            if discretization_type == 1:

                X1_con = np.asarray(
                    equipopulated_binning(output_a_con[:, 0], n_bins=n_bins), dtype=np.int32
                )
                X2_con = np.asarray(
                    equipopulated_binning(output_b_con[:, 0], n_bins=n_bins), dtype=np.int32
                )
                X1_incon = np.asarray(
                    equipopulated_binning(output_a_incon[:, 0], n_bins=n_bins), dtype=np.int32
                )
                X2_incon = np.asarray(
                    equipopulated_binning(output_b_incon[:, 0], n_bins=n_bins), dtype=np.int32
                )
            elif discretization_type == 2:
                X1_con = np.asarray(
                    equispaced_binning(output_a_con[:, 0], n_bins=n_bins), dtype=np.int32
                )
                X2_con = np.asarray(
                    equispaced_binning(output_b_con[:, 0], n_bins=n_bins), dtype=np.int32
                )
                X1_incon = np.asarray(
                    equispaced_binning(output_a_incon[:, 0], n_bins=n_bins), dtype=np.int32
                )
                X2_incon = np.asarray(
                    equispaced_binning(output_b_incon[:, 0], n_bins=n_bins), dtype=np.int32
                )
            else:
                raise NotImplementedError

            Y_con = np.asarray(target_con, dtype=np.int32)
            Y_incon = np.asarray(target_incon, dtype=np.int32)

            information_array[iteration, 0, :] = compute_information_metrics(  # type: ignore
                Y_con, X1_con, X2_con
            )

            information_array[iteration, 1, :] = compute_information_metrics(  # type: ignore
                Y_incon, X1_incon, X2_incon
            )
        temp["information"] = information_array
        recomputed_data[animal_id] = temp

    return recomputed_data


def process_file(filename, n_bins):

    try:
        data = load_pickle(filename)
        recomputed_data = recompute(data, n_bins)
        epoch = config["epoch"]
        region_name = filename.rsplit(f"_{epoch}")[0].rsplit("_")[-1]
        if config["discretize_decoding"] == 1:
            decoding = "equipopulated"
        elif config["discretize_decoding"] == 2:
            decoding = "equispaced"
        with open(
            f"./data/generated/recomputed_{region_name}_{epoch}_{decoding}_{n_bins}.pkl", "wb"
        ) as f:
            pkl.dump(recomputed_data, f)
        return (True, f"Processed: {region_name}")
    except Exception as e:
        return (False, f"FAILED {filename}: {str(e)}")


def save_recomputed_data_parallel(files, n_bins):

    total_cores = os.cpu_count()

    workers = total_cores // 2  # type: ignore
    results = Parallel(n_jobs=workers)(delayed(process_file)(f, n_bins) for f in tqdm(files))

    failures = [msg for success, msg in results if not success]  # type: ignore

    if failures:
        print(f"Process completed with {len(failures)} errors:")
        for fail_msg in failures:
            print(fail_msg)
    else:
        print("Success: All files processed without errors.")


def collapse_animal(animal):
    # the 7 is mia, mib, tvmi, unqa, unqb, red, syn
    rsi_congruent = np.nanmean(
        animal["information"][:, 0, 2]
        - (animal["information"][:, 0, 0] + animal["information"][:, 0, 0]),
        axis=0,
    )
    rsi_incongruent = np.nanmean(
        animal["information"][:, 1, 2]
        - (animal["information"][:, 1, 0] + animal["information"][:, 1, 0]),
        axis=0,
    )
    synergy = np.nanmean(animal["information"][:, :, 6], axis=0)
    redundancy = np.nanmean(animal["information"][:, :, 5], axis=0)

    return rsi_congruent, rsi_incongruent, synergy, redundancy


def pids_per_region(data):
    rsi_congruent_array = []
    rsi_incongruent_array = []
    redundancy_array = []
    synergy_array = []
    for eid in data.keys():
        rsi_congruent, rsi_incongruent, synergy, redundancy = collapse_animal(data[eid])
        rsi_congruent_array.append(rsi_congruent)
        rsi_incongruent_array.append(rsi_incongruent)
        redundancy_array.append(redundancy)
        synergy_array.append(synergy)
    return (
        np.asarray(rsi_congruent_array),
        np.asarray(rsi_incongruent_array),
        np.asarray(redundancy_array),
        np.asarray(synergy_array),
    )


def p_value_check(p_value):
    if p_value < 0.001:  # type: ignore
        p_value_text = "***"
    elif p_value < 0.01:  # type: ignore
        p_value_text = "**"
    elif p_value < 0.05:  # type: ignore
        p_value_text = "*"
    else:
        p_value_text = "n.s."  # Not significant
    return p_value_text


def individual_region_means(data):
    rsi_congruent_array, rsi_incongruent_array, redundancy_array, synergy_array = pids_per_region(
        data
    )
    rsi_congruent = np.nanmean(rsi_congruent_array, axis=0)
    rsi_incongruent = np.nanmean(rsi_incongruent_array, axis=0)
    redundancy = np.nanmean(redundancy_array, axis=0)
    synergy = np.nanmean(synergy_array, axis=0)

    return rsi_congruent, rsi_incongruent, redundancy, synergy


def individual_region_sems(data):
    rsi_congruent_array, rsi_incongruent_array, redundancy_array, synergy_array = pids_per_region(
        data
    )
    rsi_congruent = np.nanstd(rsi_congruent_array, axis=0) / np.sqrt(len(rsi_congruent_array))
    rsi_incongruent = np.nanstd(rsi_incongruent_array, axis=0) / np.sqrt(
        len(rsi_incongruent_array)
    )
    redundancy = np.nanstd(redundancy_array, axis=0) / np.sqrt(len(redundancy_array))
    synergy = np.nanstd(synergy_array, axis=0) / np.sqrt(len(synergy_array))

    return rsi_congruent, rsi_incongruent, redundancy, synergy


def congregate_data(files):

    region_names = []
    rsi_congruent = []
    rsi_incongruent = []
    redundancy = []
    synergy = []
    for filename in files:
        with open(filename, "rb") as f:
            data = pkl.load(f)

        if data == {}:
            continue

        region_name = filename.rsplit("_stim")[0].rsplit("_")[-1]
        region_names.append(region_name)

        rsi_congruent_means, rsi_incongruent_means, redundancy_means, synergy_means = (
            individual_region_means(data)
        )
        rsi_congruent_sems, rsi_incongruent_sems, redundancy_sems, synergy_sems = (
            individual_region_sems(data)
        )

        rsi_congruent.append([rsi_congruent_means, rsi_congruent_sems])
        rsi_incongruent.append([rsi_incongruent_means, rsi_incongruent_sems])
        redundancy.append([redundancy_means, redundancy_sems])
        synergy.append([synergy_means, synergy_sems])

    return (
        np.asarray(rsi_congruent),
        np.asarray(rsi_incongruent),
        np.asarray(redundancy),
        np.asarray(synergy),
        region_names,
    )


def plot_all_rsis(rsi_congruent, rsi_incongruent):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.despine()
    a = np.mean(rsi_incongruent, axis=0)[0]
    b = np.mean(rsi_congruent, axis=0)[0]
    c = np.std(rsi_incongruent, axis=0)[0] / np.sqrt(len(rsi_incongruent))
    d = np.std(rsi_congruent, axis=0)[0] / np.sqrt(len(rsi_congruent))
    ax.bar(
        np.arange(2),
        [a, b],
        edgecolor="k",
        color=["#FF4D4D", "#4D79FF"],
        yerr=[c, d],
        capsize=2,
    )
    r, p_value = wilcoxon(rsi_incongruent[:, 0], rsi_congruent[:, 0])
    if p_value < 0.001:  # type: ignore
        p_value_text = "***"
    elif p_value < 0.01:  # type: ignore
        p_value_text = "**"
    elif p_value < 0.05:  # type: ignore
        p_value_text = "*"
    else:
        p_value_text = "n.s."  # Not significant
    x1, x2 = 0, 1  # 0 and 1 correspond to the index of the bars
    line_y = np.max([a, b]) + 1.5 * np.max([c, d])  # Position the line above the taller bar

    # Plot a horizontal line
    plt.plot([x1, x2], [line_y, line_y], color="black", linewidth=1)

    # Plot the vertical ticks at the ends of the line

    # 3. Display the p-value text
    # Set the position for the text
    text_x = (x1 + x2) / 2
    text_y = line_y
    plt.text(text_x, text_y, p_value_text, ha="center", va="bottom", fontsize=12)
    ax.set_xticks(np.arange(2), ["Incongruent", "Congruent"])
    ax.set_ylabel("RSI")


def plot_all_decompositions(redundancy, synergy):
    fig, ax = plt.subplots(figsize=(8, 8))
    redundancy_means = np.mean(redundancy[:, 0, :], axis=0)
    redundancy_sems = np.std(redundancy[:, 0, :], axis=0) / np.sqrt(len(redundancy))

    synergy_means = np.mean(synergy[:, 0, :], axis=0)
    synergy_sems = np.std(synergy[:, 0, :], axis=0) / np.sqrt(len(synergy))

    ax.bar(
        np.arange(2),
        [redundancy_means[1], synergy_means[1]],
        yerr=[redundancy_sems[1], synergy_sems[1]],
        edgecolor="k",
        color=["#FF4D4D", "#EB9C9C"],
        width=0.4,
        label="Incongruent",
    )

    ax.bar(
        np.arange(2) + 0.4,
        [redundancy_means[0], synergy_means[0]],
        yerr=[redundancy_sems[0], synergy_sems[0]],
        edgecolor="k",
        color=["#4D79FF", "#7C9AF3"],
        width=0.4,
        label="Congruent",
    )
    sns.despine()
    r, p_value_red = wilcoxon(redundancy[:, 0, 0], redundancy[:, 0, 1])
    r, p_value_syn = wilcoxon(synergy[:, 0, 0], synergy[:, 0, 1])

    p_red_text = p_value_check(p_value_red)
    p_syn_text = p_value_check(p_value_syn)

    x1, x2 = 0, 0.375
    x3, x4 = 1, 1.375
    line_red = np.max([redundancy_means[0], redundancy_means[1]]) + 1.5 * np.max(
        [redundancy_sems[0], redundancy_sems[1]]
    )
    line_syn = np.max([synergy_means[0], synergy_means[1]]) + 1.5 * np.max(
        [synergy_sems[0], synergy_sems[1]]
    )

    plt.plot([x1, x2], [line_red, line_red], color="black", linewidth=1)
    text_x = (x1 + x2) / 2
    plt.text(text_x, line_red, p_red_text, ha="center", va="bottom", fontsize=12)

    plt.plot([x3, x4], [line_syn, line_syn], color="black", linewidth=1)
    text_x = (x3 + x4) / 2
    plt.text(text_x, line_syn, p_syn_text, ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("Information (in bits)")
    ax.legend()
    ax.set_xticks(np.arange(2) + 0.2, ["Redundancy", "Synergy"])


def plot_regions_rsi(rsi_incongruent, rsi_congruent, region_names):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.despine()

    ax.bar(
        np.arange(len(rsi_incongruent)),
        rsi_incongruent[:, 0],
        yerr=rsi_incongruent[:, 1],
        edgecolor="k",
        color="#FF4D4D",
        width=0.4,
        label="incongruent",
    )
    ax.bar(
        np.arange(len(rsi_congruent)) + 0.4,
        rsi_congruent[:, 0],
        yerr=rsi_congruent[:, 1],
        edgecolor="k",
        color="#4D79FF",
        width=0.4,
        label="congruent",
    )

    ax.set_xticks(np.arange(len(region_names)), region_names, rotation=90)
    ax.set_ylabel("RSI")
    ax.legend()


if __name__ == "__main__":

    location = config["recompute_location"]
    files = glob(f"{location}/*.pkl")
    n_bins = config["n_bins_decoding"]
    save_recomputed_data_parallel(files, n_bins)
