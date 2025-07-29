from pathlib import Path
from typing import Tuple, Union
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from glob import glob
from scipy.stats import wilcoxon
from matplotlib.container import BarContainer


def compute_means_and_sems(all_values, congruent, incongruent):

    all_mean = np.mean(all_values, axis=0)
    congruent_mean = np.mean(congruent, axis=0)
    incongruent_mean = np.mean(incongruent, axis=0)
    # middling_mean = np.mean(middling, axis=0)

    all_sem = np.std(all_values, axis=0) / np.sqrt(all_values.shape[0])
    congruent_sem = np.std(congruent, axis=0) / np.sqrt(congruent.shape[0])
    incongruent_sem = np.std(incongruent, axis=0) / np.sqrt(incongruent.shape[0])

    return np.asarray([all_mean, congruent_mean, incongruent_mean]), np.asarray(
        [all_sem, congruent_sem, incongruent_sem]
    )


def convert_to_markers(unique, red, syn):
    markers = []
    if unique < 0.001:
        markers.append("***")
    elif unique < 0.01:
        markers.append("**")
    elif unique < 0.05:
        markers.append("*")
    else:
        markers.append("n.s.")

    if red < 0.001:
        markers.append("***")
    elif red < 0.01:
        markers.append("**")
    elif red < 0.05:
        markers.append("*")
    else:
        markers.append("n.s.")

    if syn < 0.001:
        markers.append("***")
    elif syn < 0.01:
        markers.append("**")
    elif syn < 0.05:
        markers.append("*")
    else:
        markers.append("n.s.")

    return markers


def aggregate_data(region_data, congruent_id="subsampled"):
    trials = []
    all_pid = []
    subsampled_congruent_pid = []
    incongruent_pid = []
    all_joint = []
    subsampled_congruent_joint = []
    incongruent_joint = []
    neurons = []
    for eid in region_data.keys():
        neurons.append(region_data[eid]["neurons"])
        trials.append(region_data[eid]["trials"])
        all_pid.append(region_data[eid]["all"]["pid"])
        subsampled_congruent_pid.append(region_data[eid][congruent_id]["pid"])
        incongruent_pid.append(region_data[eid]["incongruent"]["pid"])

        all_joint.append(region_data[eid]["all"]["trivariate"])
        subsampled_congruent_joint.append(region_data[eid][congruent_id]["trivariate"])
        incongruent_joint.append(region_data[eid]["incongruent"]["trivariate"])

    trials = np.asarray(trials)
    all_pid = np.concatenate(all_pid)
    subsampled_congruent_pid = np.concatenate(subsampled_congruent_pid)
    incongruent_pid = np.concatenate(incongruent_pid)

    all_joint = np.concatenate(all_joint)
    subsampled_congruent_joint = np.concatenate(subsampled_congruent_joint)
    incongruent_joint = np.concatenate(incongruent_joint)

    neurons = np.asarray(neurons)

    return (
        trials,
        all_pid,
        subsampled_congruent_pid,
        incongruent_pid,
        all_joint,
        subsampled_congruent_joint,
        incongruent_joint,
        neurons,
    )


def aggregate_all_data(region_pickles, congruent_id="subsampled"):

    trials = []
    all_pid = []
    subsampled_congruent_pid = []
    incongruent_pid = []
    all_joint = []
    subsampled_congruent_joint = []
    incongruent_joint = []
    neurons = []

    for idx, pickle_location in enumerate(region_pickles):

        with open(pickle_location, "rb") as f:
            region_data = pkl.load(f)

        for eid in region_data.keys():
            neurons.append(region_data[eid]["neurons"])
            trials.append(region_data[eid]["trials"])
            all_pid.append(region_data[eid]["all"]["pid"])
            subsampled_congruent_pid.append(region_data[eid][congruent_id]["pid"])
            incongruent_pid.append(region_data[eid]["incongruent"]["pid"])

            all_joint.append(region_data[eid]["all"]["trivariate"])
            subsampled_congruent_joint.append(region_data[eid][congruent_id]["trivariate"])
            incongruent_joint.append(region_data[eid]["incongruent"]["trivariate"])
    trials = np.asarray(trials)
    all_pid = np.concatenate(all_pid)
    subsampled_congruent_pid = np.concatenate(subsampled_congruent_pid)
    incongruent_pid = np.concatenate(incongruent_pid)
    all_joint = np.concatenate(all_joint)
    subsampled_congruent_joint = np.concatenate(subsampled_congruent_joint)
    incongruent_joint = np.concatenate(incongruent_joint)
    neurons = np.asarray(neurons)

    return (
        trials,
        all_pid,
        subsampled_congruent_pid,
        incongruent_pid,
        all_joint,
        subsampled_congruent_joint,
        incongruent_joint,
        neurons,
    )


def regional_pid_results(region_data, region_name, zero_out=False, congruent_id="subsampled"):

    if region_name != "all":
        (
            trials,
            all_pid,
            subsampled_congruent_pid,
            incongruent_pid,
            all_joint,
            subsampled_congruent_joint,
            incongruent_joint,
            neurons,
        ) = aggregate_data(region_data, congruent_id)
    else:
        (
            trials,
            all_pid,
            subsampled_congruent_pid,
            incongruent_pid,
            all_joint,
            subsampled_congruent_joint,
            incongruent_joint,
            neurons,
        ) = aggregate_all_data(region_data, congruent_id)

    if zero_out:
        all_pid[all_pid < 0] = 0
        subsampled_congruent_pid[subsampled_congruent_pid < 0] = 0
        incongruent_pid[incongruent_pid < 0] = 0

        all_joint[all_joint < 0] = 0
        subsampled_congruent_joint[subsampled_congruent_joint < 0] = 0
        incongruent_joint[incongruent_joint < 0] = 0

    means_pid, sems_pid = compute_means_and_sems(
        all_pid,
        subsampled_congruent_pid,
        incongruent_pid,
    )

    means_joint, sems_joint = compute_means_and_sems(
        all_joint,
        subsampled_congruent_joint,
        incongruent_joint,
    )

    _, p_red = wilcoxon(subsampled_congruent_pid[:, 2], incongruent_pid[:, 2])
    _, p_syn = wilcoxon(subsampled_congruent_pid[:, 3], incongruent_pid[:, 3])
    _, p_joint = wilcoxon(subsampled_congruent_joint[:, 0], incongruent_joint[:, 0])

    significance_markers = convert_to_markers(p_joint, p_red, p_syn)

    colors = ["#a2c4e3", "#ffc080", "#9bcd9b"]  # all, congruent, incongruent

    fig, ax = plt.subplots(figsize=(16, 4), ncols=3)

    # plot red and syn
    ax[0].bar(
        np.arange(3) - 0.3,
        [means_joint[0, 0], means_pid[0, 2], means_pid[0, 3]],
        yerr=[sems_joint[0, 0], sems_pid[0, 2], sems_pid[0, 3]],
        color=colors[0],
        width=0.3,
        capsize=5,
        edgecolor="k",
        linestyle="dashed",
        label="All",
    )
    ax[0].bar(
        np.arange(3),
        [means_joint[1, 0], means_pid[1, 2], means_pid[1, 3]],
        yerr=[sems_joint[1, 0], sems_pid[1, 2], sems_pid[1, 3]],
        color=colors[1],
        width=0.3,
        capsize=5,
        edgecolor="k",
        linestyle="dashed",
        label="Congruent",
    )

    ax[0].bar(
        np.arange(3) + 0.3,
        [means_joint[2, 0], means_pid[2, 2], means_pid[2, 3]],
        yerr=[sems_joint[2, 0], sems_pid[2, 2], sems_pid[2, 3]],
        color=colors[2],
        width=0.3,
        capsize=5,
        edgecolor="k",
        linestyle="dashed",
        label="Incongruent",
    )

    ax[0].set_xticks(np.arange(3), ["Joint", "Redundancy", "Synergy"])
    for idx in range(3):
        ax[idx].spines["top"].set_visible(False)
        ax[idx].spines["right"].set_visible(False)

    if congruent_id == "congruent":
        sns.boxplot(trials, ax=ax[1], palette=colors, width=0.25)
        ax[1].set_xticks(np.arange(3), ["All", "Congruent", "Incongruent"])
    else:
        sns.boxplot(
            [trials[:, 0], trials[:, 2]], palette=[colors[0], colors[2]], ax=ax[1], width=0.25
        )
        ax[1].set_xticks(np.arange(2), ["All", "Incongruent"])

    # plot significances
    x_base = np.arange(3)
    bar_width = 0.3
    for i, x_pos in enumerate(x_base):
        x1_bar_pos = x_pos
        x2_bar_pos = x_pos + bar_width

        if i == 0:  # This corresponds to the 'unique_info' part
            mean_bar2 = means_joint[1]
            sem_bar2 = sems_joint[1]
        else:  # This corresponds to the pid_means part
            mean_bar2 = means_pid[1, i + 1]  # means_pid[1, 2] for i=1, means_pid[1,3] for i=2
            sem_bar2 = sems_pid[1, i + 1]

        # Data for the 'Incongruent' bar at current xtick `i`
        if i == 0:  # This corresponds to the 'unique_info' part
            mean_bar3 = means_joint[2]
            sem_bar3 = sems_joint[2]
        else:  # This corresponds to the means_pid part
            mean_bar3 = means_pid[2, i + 1]  # pid_means[2, 2] for i=1, pid_means[2,3] for i=2
            sem_bar3 = sems_pid[2, i + 1]

        max_y = max(mean_bar2 + sem_bar2, mean_bar3 + sem_bar3)
        y_sig_line = max_y + 0.001  # Adjust this offset as needed
        y_sig_text = y_sig_line  # Adjust this offset as needed

        marker = significance_markers[i]
        ax[0].plot([x1_bar_pos, x2_bar_pos], [y_sig_line, y_sig_line], "k-")  # Horizontal line
        ax[0].text(
            (x1_bar_pos + x2_bar_pos) / 2,
            y_sig_text,
            marker,
            ha="center",
            va="bottom",
            color="k",
            fontsize=12,
        )

    ax[2].boxplot(neurons)
    ax[2].set_xticklabels(["Neurons"])

    ax[0].set_ylabel("Bits")
    ax[1].set_ylabel("Trials")
    ax[2].set_ylabel("Count")

    plt.suptitle(f"{region_name}")

    plt.tight_layout()

    if zero_out:
        info = "zeroed"
    else:
        info = "bias_corrected"

    filelocation = Path(__file__).parent.parent.joinpath(
        f"reports/figures/region_pid/{region_name}_{congruent_id}_{info}.png"
    )
    plt.savefig(filelocation, bbox_inches="tight", facecolor="white")
    plt.close()

    synergy_delta = means_pid[1, 3] - means_pid[2, 3]
    redundancy_delta = means_pid[1, 2] - means_pid[2, 2]

    # RSI index

    rsi_congruent = means_pid[1, 2] - means_pid[1, 3]
    rsi_incongruent = means_pid[2, 2] - means_pid[2, 3]

    return synergy_delta, redundancy_delta, rsi_congruent, rsi_incongruent
