import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from glob import glob


def plot_subsample(ax, data, original_values, pid_array=None):  # very hacky, change this

    colors = ["#a2c4e3", "#ffc080", "#9bcd9b"]  # all, congruent, incongruent
    if pid_array is None:
        eids = list(data.keys())
        pid_array = []
        for eid in eids:
            for repeats in range(0, 5):
                temp = data[eid][repeats]["pid"]
                if len(temp) != 0:
                    pid_array.append(temp)
        pid_array = np.concatenate(pid_array)

    pid_means = np.mean(pid_array, axis=0)
    pid_sems = np.std(pid_array, axis=0) / np.sqrt(len(pid_array))

    # fig, ax = plt.subplots(figsize=(4, 4), ncols=1)
    ax.bar(
        np.arange(0, 2),
        pid_means[2:],
        yerr=pid_sems[2:],
        color="#ffaa80",
        edgecolor="k",
        capsize=5,
        linestyle="dashed",
        width=0.4,
        label="subsampled-congruent",
    )
    ax.bar(
        np.arange(0, 2) + 0.4,
        [original_values[0], original_values[1]],
        yerr=[original_values[2], original_values[3]],
        color="#9bcd9b",
        edgecolor="k",
        capsize=5,
        linestyle="dashed",
        width=0.4,
        label="original-incongruent",
    )
    ax.legend()
    ax.set_xticks(np.arange(0, 2) + 0.2, ["Redundant", "Synergy"])
    ax.set_title(f"Subsampled Congruent Comparison")
    return ax


def collate_data_for_region(region_data):
    all_mutual = []
    congruent_mutual = []
    incongruent_mutual = []
    # middling_mutual = []

    all_pid = []
    congruent_pid = []
    incongruent_pid = []

    all_joint = []
    congruent_joint = []
    incongruent_joint = []

    eids = list(region_data.keys())
    for e in eids:
        mouse_data = region_data[e]
        all_mutual.append(mouse_data["all"]["mutual_information"])
        congruent_mutual.append(mouse_data["congruent"]["mutual_information"])
        incongruent_mutual.append(mouse_data["incongruent"]["mutual_information"])
        # middling_mutual.append(mouse_data["middling_incongruent"]["mutual_information"])

        # check if empty else append
        if mouse_data["all"]["pid"].shape == (0,):
            continue
        else:
            all_pid.append(mouse_data["all"]["pid"])
            all_joint.append(mouse_data["all"]["tvmi"])

        if mouse_data["congruent"]["pid"].shape == (0,):
            continue
        else:
            congruent_pid.append(mouse_data["congruent"]["pid"])
            congruent_joint.append(mouse_data["congruent"]["tvmi"])

        if mouse_data["incongruent"]["pid"].shape == (0,):
            continue
        else:
            incongruent_pid.append(mouse_data["incongruent"]["pid"])
            incongruent_joint.append(mouse_data["incongruent"]["tvmi"])

    all_mutual = np.concatenate(all_mutual)
    congruent_mutual = np.concatenate(congruent_mutual)
    incongruent_mutual = np.concatenate(incongruent_mutual)

    all_pid = np.concatenate(all_pid)
    congruent_pid = np.concatenate(congruent_pid)
    incongruent_pid = np.concatenate(incongruent_pid)

    all_joint = np.concatenate(all_joint)
    congruent_joint = np.concatenate(congruent_joint)
    incongruent_joint = np.concatenate(incongruent_joint)

    return (
        all_mutual,
        congruent_mutual,
        incongruent_mutual,
        all_pid,
        congruent_pid,
        incongruent_pid,
        all_joint,
        congruent_joint,
        incongruent_joint,
    )


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


# compute proportions correctly
def compute_proportions(
    all_values, congruent, incongruent, joint_all_values, joint_congruent, joint_incongruent
):
    pass


def plot_information(
    all_mutual,
    congruent_mutual,
    incongruent_mutual,
    all_pid,
    congruent_pid,
    incongruent_pid,
    trial_data_aggregate,
    region_name,
    region_flag,
    all_joint,
    congruent_joint,
    incongruent_joint,
    subsample_data=None,
    plt_jt=False,
    normalize_by_joint=False,
    subsampled_type=0,
):

    # i think changing this makes more sense
    # have only synergy and redundancy for all three conditions
    # collapse the mutual information plot into the pid plot
    # have trials separately

    colors = ["#a2c4e3", "#ffc080", "#9bcd9b"]  # all, congruent, incongruent

    if region_flag:
        text_color = "black"
    else:
        text_color = "red"

    mutual_info_means, mutual_info_sems = compute_means_and_sems(
        all_mutual, congruent_mutual, incongruent_mutual
    )
    pid_means, pid_sems = compute_means_and_sems(all_pid, congruent_pid, incongruent_pid)
    joint_means, joint_sems = compute_means_and_sems(all_joint, congruent_joint, incongruent_joint)

    # let's not plot the joints for now
    if (subsample_data is not None) and (not normalize_by_joint):
        fig, ax = plt.subplots(figsize=(18, 5), ncols=3)
        ax[2].sharey(ax[0])
    else:

        fig, ax = plt.subplots(figsize=(18, 5), ncols=2)

    # first one plots pid values
    if (subsample_data is not None) and (not normalize_by_joint):
        ax_num = 3
    else:
        ax_num = 2

    for idx in range(ax_num):
        ax[idx].spines["right"].set_visible(False)
        ax[idx].spines["top"].set_visible(False)

    if normalize_by_joint:
        pid_means /= np.sum(pid_means, axis=1)[:, np.newaxis]
        # mutual_info_means /= np.sum(pid_means)
        # don't use mutual info means but rather pid_means
        # overriding mutual info means.
        # very ugly
        mutual_info_means[0] = np.mean(pid_means[0, 0:2])
        mutual_info_means[1] = np.mean(pid_means[1, 0:2])
        mutual_info_means[2] = np.mean(pid_means[2, 0:2])

    ax[0].bar(
        np.arange(3) - 0.3,
        [mutual_info_means[0], pid_means[0, 2], pid_means[0, 3]],
        yerr=[mutual_info_sems[0], pid_sems[0, 2], pid_sems[0, 3]],
        color=colors[0],
        width=0.3,
        capsize=5,
        edgecolor="k",
        linestyle="dashed",
        label="All",
    )
    ax[0].bar(
        np.arange(3),
        [mutual_info_means[1], pid_means[1, 2], pid_means[1, 3]],
        yerr=[mutual_info_sems[1], pid_sems[1, 2], pid_sems[1, 3]],
        color=colors[1],
        width=0.3,
        capsize=5,
        edgecolor="k",
        linestyle="dashed",
        label="Congruent",
    )
    ax[0].bar(
        np.arange(3) + 0.3,
        [mutual_info_means[2], pid_means[2, 2], pid_means[2, 3]],
        yerr=[mutual_info_sems[2], pid_sems[2, 2], pid_sems[2, 3]],
        color=colors[2],
        width=0.3,
        capsize=5,
        edgecolor="k",
        linestyle="dashed",
        label="Incongruent",
    )

    # now for trial data
    sns.barplot(
        trial_data_aggregate[:, 0:3],
        ax=ax[1],
        capsize=0.1,
        edgecolor="k",
        linewidth=1,
        palette=colors,
        err_kws={"linewidth": 1},
    )
    ax[0].set_xticks(np.arange(0, 3), ["Unique", "Redundant", "Synergy"])
    if normalize_by_joint:
        ax[0].set_ylabel("Fraction of joint")
    else:
        ax[0].set_ylabel("PID")

    ax[1].set_xticks(np.arange(0, 3), ["All", "Congruent", "Incongruent"])
    ax[1].set_ylabel("Trials")

    ax[0].set_title("Partial information decomposition")
    ax[0].legend()
    ax[1].set_title("Trials used")
    plt.suptitle(f"{region_name}", color=text_color)

    ## subsampled stuff
    if (subsample_data is not None) and (not normalize_by_joint):
        original_incongruent_values = [
            pid_means[2, 2],
            pid_means[2, 3],
            pid_sems[2, 2],
            pid_sems[2, 3],
        ]
        if subsampled_type == 0:
            plot_subsample(ax[2], subsample_data, original_incongruent_values)
        else:
            plot_subsample(ax[2], None, original_incongruent_values, subsample_data)

    info = ""
    if normalize_by_joint:
        info = "norm"

    plt.savefig(
        f"../reports/figures/region_pid/{region_name}_{info}.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def plot_each_region(
    region_data, trial_data, region_name, region_flag, subsample_data=None, normalize=False
):

    (
        all_mutual,
        congruent_mutual,
        incongruent_mutual,
        all_pid,
        congruent_pid,
        incongruent_pid,
        all_joint,
        congruent_joint,
        incongruent_joint,
    ) = collate_data_for_region(region_data)

    # print(f"{region_name}: {all_pid.shape}, {congruent_pid.shape}, {incongruent_pid.shape}")

    # trial_data is a dict with keys eid and then conditions
    # how to plot this
    trial_data_aggregate = []

    for eid in trial_data.keys():
        trial_data_aggregate.append(trial_data[eid])

    trial_data_aggregate = np.asarray(trial_data_aggregate)
    plot_information(
        all_mutual,
        congruent_mutual,
        incongruent_mutual,
        all_pid,
        congruent_pid,
        incongruent_pid,
        trial_data_aggregate,
        region_name,
        region_flag,
        all_joint,
        congruent_joint,
        incongruent_joint,
        subsample_data,
        plt_jt=False,
        normalize_by_joint=normalize,
    )


def check_decompostion_results_collate(data):

    eids = list(data.keys())

    all_pid = []
    c_pid = []
    ic_pid = []

    all_mi = []
    c_mi = []
    ic_mi = []

    all_joint = []
    c_joint = []
    ic_joint = []

    for eid in eids:
        if len(data[eid]["all"]["pid"]) != 0:
            all_pid.append(data[eid]["all"]["pid"])
        if len(data[eid]["congruent"]["pid"]) != 0:
            c_pid.append(data[eid]["congruent"]["pid"])
        if len(data[eid]["incongruent"]["pid"]) != 0:
            ic_pid.append(data[eid]["incongruent"]["pid"])

        if len(data[eid]["all"]["mutual_information"]) != 0:
            all_mi.append(data[eid]["all"]["mutual_information"])
        if len(data[eid]["congruent"]["mutual_information"]) != 0:
            c_mi.append(data[eid]["congruent"]["mutual_information"])
        if len(data[eid]["incongruent"]["mutual_information"]) != 0:
            ic_mi.append(data[eid]["incongruent"]["mutual_information"])

        if len(data[eid]["all"]["tvmi"]) != 0:
            all_joint.append(data[eid]["all"]["tvmi"])
        if len(data[eid]["congruent"]["tvmi"]) != 0:
            c_joint.append(data[eid]["congruent"]["tvmi"])
        if len(data[eid]["incongruent"]["tvmi"]) != 0:
            ic_joint.append(data[eid]["incongruent"]["tvmi"])

    all_pid = np.concatenate(all_pid)
    c_pid = np.concatenate(c_pid)
    ic_pid = np.concatenate(ic_pid)

    all_mi = np.concatenate(all_mi)
    c_mi = np.concatenate(c_mi)
    ic_mi = np.concatenate(ic_mi)

    all_joint = np.concatenate(all_joint)
    c_joint = np.concatenate(c_joint)
    ic_joint = np.concatenate(ic_joint)

    all_difference = all_joint - np.sum(all_pid, axis=1)
    c_difference = c_joint - np.sum(c_pid, axis=1)
    ic_difference = ic_joint - np.sum(ic_pid, axis=1)

    all_difference_mean = np.mean(all_difference)
    all_difference_sem = np.std(all_difference) / np.sqrt(all_difference.shape[0])

    c_difference_mean = np.mean(c_difference)
    c_difference_sem = np.std(c_difference) / np.sqrt(c_difference.shape[0])

    ic_difference_mean = np.mean(ic_difference)
    ic_difference_sem = np.std(ic_difference) / np.sqrt(ic_difference.shape[0])

    all_mi_mean = np.mean(all_mi)
    all_mi_sem = np.std(all_mi) / np.sqrt(all_mi.shape[0])

    c_mi_mean = np.mean(c_mi)
    c_mi_sem = np.std(c_mi) / np.sqrt(c_mi.shape[0])

    ic_mi_mean = np.mean(ic_mi)
    ic_mi_sem = np.std(ic_mi) / np.sqrt(ic_mi.shape[0])

    all_joint_mean = np.mean(all_joint)
    all_joint_sem = np.std(all_joint) / np.sqrt(all_joint.shape[0])

    c_joint_mean = np.mean(c_joint)
    c_joint_sem = np.std(c_joint) / np.sqrt(c_joint.shape[0])

    ic_joint_mean = np.mean(ic_joint)
    ic_joint_sem = np.std(ic_joint) / np.sqrt(ic_joint.shape[0])

    all_pid_synergy = all_pid[:, 3]
    c_pid_synergy = c_pid[:, 3]
    ic_pid_synergy = ic_pid[:, 3]

    all_pid_red = all_pid[:, 2]
    c_pid_red = c_pid[:, 2]
    ic_pid_red = ic_pid[:, 2]

    all_pid_unq = all_pid[:, 0:2]
    c_pid_unq = c_pid[:, 0:2]
    ic_pid_unq = ic_pid[:, 0:2]

    # now we compute means and sems

    all_pid_synergy_mean = np.mean(all_pid_synergy)
    all_pid_synergy_sem = np.std(all_pid_synergy) / np.sqrt(all_pid_synergy.shape[0])

    c_pid_synergy_mean = np.mean(c_pid_synergy)
    c_pid_synergy_sem = np.std(c_pid_synergy) / np.sqrt(c_pid_synergy.shape[0])

    ic_pid_synergy_mean = np.mean(ic_pid_synergy)
    ic_pid_synergy_sem = np.std(ic_pid_synergy) / np.sqrt(ic_pid_synergy.shape[0])

    all_pid_red_mean = np.mean(all_pid_red)
    all_pid_red_sem = np.std(all_pid_red) / np.sqrt(all_pid_red.shape[0])

    c_pid_red_mean = np.mean(c_pid_red)
    c_pid_red_sem = np.std(c_pid_red) / np.sqrt(c_pid_red.shape[0])

    ic_pid_red_mean = np.mean(ic_pid_red)
    ic_pid_red_sem = np.std(ic_pid_red) / np.sqrt(ic_pid_red.shape[0])

    all_pid_unq_mean = np.mean(all_pid_unq)
    all_pid_unq_sem = np.std(all_pid_unq) / np.sqrt(all_pid_unq.shape[0])

    c_pid_unq_mean = np.mean(c_pid_unq)
    c_pid_unq_sem = np.std(c_pid_unq) / np.sqrt(c_pid_unq.shape[0])

    ic_pid_unq_mean = np.mean(ic_pid_unq)
    ic_pid_unq_sem = np.std(ic_pid_unq) / np.sqrt(ic_pid_unq.shape[0])

    # now we return things in the proper order
    # unique information : all, cong, incong (3) x 2 (mean, sem)
    # synergy information : all, cong, incong (3) x 2 (mean, sem)
    # redundant information : all, cong, incong (3) x 2 (mean, sem)
    # mutual information and joint information : all, cong, incong (3) x 2 (mean, sem)
    # difference

    unique_information = np.zeros((3, 2))  # type: ignore
    synergy_information = np.zeros((3, 2))
    redundant_information = np.zeros((3, 2))
    mutual_information = np.zeros((3, 2))
    joint_information = np.zeros((3, 2))
    difference = np.zeros((3, 2))

    unique_information[:, 0] = [all_pid_unq_mean, c_pid_unq_mean, ic_pid_unq_mean]
    unique_information[:, 1] = [all_pid_unq_sem, c_pid_unq_sem, ic_pid_unq_sem]

    synergy_information[:, 0] = [all_pid_synergy_mean, c_pid_synergy_mean, ic_pid_synergy_mean]
    synergy_information[:, 1] = [all_pid_synergy_sem, c_pid_synergy_sem, ic_pid_synergy_sem]

    redundant_information[:, 0] = [all_pid_red_mean, c_pid_red_mean, ic_pid_red_mean]
    redundant_information[:, 1] = [all_pid_red_sem, c_pid_red_sem, ic_pid_red_sem]

    mutual_information[:, 0] = [all_mi_mean, c_mi_mean, ic_mi_mean]
    mutual_information[:, 1] = [all_mi_sem, c_mi_sem, ic_mi_sem]

    joint_information[:, 0] = [all_joint_mean, c_joint_mean, ic_joint_mean]
    joint_information[:, 1] = [all_joint_sem, c_joint_sem, ic_joint_sem]

    difference[:, 0] = [all_difference_mean, c_difference_mean, ic_difference_mean]
    difference[:, 1] = [all_difference_sem, c_difference_sem, ic_difference_sem]
    return (
        unique_information,
        synergy_information,
        redundant_information,
        mutual_information,
        joint_information,
        difference,
    )


def compare_plot(
    unique_information,
    synergy_information,
    redundant_information,
    mutual_information,
    joint_information,
    difference,
    region,
):

    # plt

    fig, ax = plt.subplots(ncols=5, figsize=(23, 4), sharey=True, sharex=True)

    ax[0].bar(np.arange(3), unique_information[:, 0], yerr=unique_information[:, 1])
    ax[0].set_title("Unique Information")

    ax[1].bar(np.arange(3), synergy_information[:, 0], yerr=synergy_information[:, 1])
    ax[1].set_title("Synergy Information")

    ax[2].bar(np.arange(3), redundant_information[:, 0], yerr=redundant_information[:, 1])
    ax[2].set_title("Redundant Information")

    ax[3].bar(np.arange(3), mutual_information[:, 0], yerr=mutual_information[:, 1])
    ax[3].set_title("Mutual Information")

    # difference = joint_information - unique_information[:, 0] - synergy_information[:, 0] - redundant_information[
    #     :,0
    # ]
    ax[4].bar(np.arange(3), difference[:, 0], yerr=difference[:, 1])

    ax[4].set_title("Joint Information delta")
    ax[0].set_xticks(np.arange(3), ["All", "Congruent", "Incongruent"])
    plt.suptitle(f"Region={region}")
    plt.savefig(
        f"../reports/figures/comparisons/{region}.png", bbox_inches="tight", facecolor="white"
    )
    plt.close()


def check_all(region_pickles, region_names):

    for idx, pickle_location in enumerate(region_pickles):

        with open(pickle_location, "rb") as f:
            region_data = pkl.load(f)

        region_name = region_names[idx]
        (
            unique_information,
            synergy_information,
            redundant_information,
            mutual_information,
            joint_information,
            difference,
        ) = check_decompostion_results_collate(region_data)

        compare_plot(
            unique_information,
            synergy_information,
            redundant_information,
            mutual_information,
            joint_information,
            difference,
            region_name,
        )
        # on average, unique information lines up with mutual information
        # sum lines up with joint information
        break
        # don't run cause it crashes weirdly


def convert_into_struct(data):
    transform_data = {}

    for k in data.keys():
        transform_data[k] = data[k]["subsampled"]

    return transform_data
