import numpy as np
import scipy.linalg
import torch


import pandas as pd
from matplotlib import pyplot as plt
import pickle as pkl

from ibl_info.broja_pid import compute_pid, coinformation, compute_pid_unbiased, unbiasedMI, MI
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
import itertools
import seaborn as sns
from glob import glob
import re
from joblib import Parallel, delayed

## load one session, and organize everything in one function


def load_session(path):
    with open(path, "rb") as f:
        rnn_model = pkl.load(f)
    return rnn_model["session_data"]


def freeman_daconis_bin_size(signal):
    """
    Chooses bin size based on the IQR of the data

    Args:
        signal (np.array): 1xtrials

    Returns:
        bin_size
    """
    iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
    bin_width = 2 * iqr / (len(signal) ** (1 / 3))
    num_bins_fd = int((signal.max() - signal.min()) / bin_width)
    return num_bins_fd


def equipopulated_binning(signal, n_bins=6):
    """
    Discretize the hidden state into equipopulated bins

    Args:
        signal (np.array): signal, 1 x trials
        n_bins (int): number of bins

    """

    discrete_data = np.zeros_like(signal)
    # discretize per recorded neuron

    discrete_row, bin_edges_p = pd.qcut(
        signal, q=n_bins, labels=False, duplicates="drop", retbins=True
    )
    discrete_data = discrete_row

    discrete_data = np.nan_to_num(discrete_data, nan=0)
    return discrete_data


def probability_binning(data, n_bins=10):
    """This bins a probability signal into given number of bins, and the binning is always equispaced, not equiprobable

    Args:
        data (trials x 1): output for n number of trials
        n_bins (int, optional): number of bins to consider. Defaults to 10.
    """

    bin_edges = np.linspace(0, 1, n_bins + 1)

    bins = np.digitize(data, bin_edges)
    return bins


def hidden_layer_binning(hidden_layer_activity, n_bins=50):
    """
    Bins the activity of the hidden layer into bins for information computation

    Args:
        hidden_layer_activity (trials x neurons): np.array of hidden layer activity
        n_bins (int, optional): number of bins. Defaults to 50.
    """
    # in general, the nonlinearity in the hidden layer in tanh, so we also use a equispaced, not equiprobable binning

    n_trials, n_neurons = hidden_layer_activity.shape
    bin_edges = np.linspace(-1, 1, n_bins + 1)

    binned_data = np.zeros_like(hidden_layer_activity)

    for neuron in range(n_neurons):

        activity = hidden_layer_activity[:, neuron]
        bins = np.digitize(activity, bin_edges)
        binned_data[:, neuron] = bins

    return binned_data


def optimal_bayesian(sides, geometric_p):
    """optimal bayesian prior given a geometric prior that determines the length of each block

    Args:
        sides (np.array): side of stimulus presentation
        geometric_p (np.float): p for geometric distribution
    """

    sides = torch.from_numpy(sides)

    # changed parameters
    lb, ub, gamma = 20, 60, 0.8
    tau = -(1 / np.log(1 - geometric_p))
    nb_blocklengths = 60
    nb_typeblocks = 2

    eps = torch.tensor(1e-15)

    alpha = torch.zeros([sides.shape[-1], nb_blocklengths, nb_typeblocks])
    alpha[0, 0, 0] = alpha[0, 0, 1] = 1 / 2  # could be either block

    alpha = alpha.reshape(-1, nb_blocklengths * nb_typeblocks)
    h = torch.zeros([nb_typeblocks * nb_blocklengths])

    # build transition matrix
    b = torch.zeros([nb_blocklengths, nb_typeblocks, nb_typeblocks])
    b[1:][:, 0, 0], b[1:][:, 1, 1] = 1, 1  # case when l_t > 0
    b[0][0][-1], b[0][-1][0] = 1, 1

    n = torch.arange(1, nb_blocklengths + 1)
    ref = torch.exp(-n / tau) * (lb <= n) * (ub >= n)
    torch.flip(ref.double(), (0,))
    hazard = torch.cummax(
        ref / torch.flip(torch.cumsum(torch.flip(ref.double(), (0,)), 0) + eps, (0,)), 0
    )[0]
    l_mat = torch.cat(
        (
            torch.unsqueeze(hazard, -1),
            torch.cat(
                (torch.diag(1 - hazard[:-1]), torch.zeros(nb_blocklengths - 1)[None]), axis=0
            ),
        ),
        axis=-1,
    )  # l_{t-1}, l_t
    transition = eps + torch.transpose(l_mat[:, :, None, None] * b[None], 1, 2).reshape(
        nb_typeblocks * nb_blocklengths, -1
    )

    # likelihood
    lks = torch.hstack(
        [
            gamma * (sides[:, None] == -1) + (1 - gamma) * (sides[:, None] == 1),
            gamma * (sides[:, None] == 1) + (1 - gamma) * (sides[:, None] == -1),
        ]
    )
    # before not equal 0, update only when the model takes an action
    # now it is 2, so never, so always update
    # just always update
    to_update = torch.unsqueeze(torch.unsqueeze(sides.not_equal(2), -1), -1) * 1

    for i_trial in range(sides.shape[-1]):
        # save priors
        if i_trial >= 0:
            if i_trial > 0:
                alpha[i_trial] = torch.sum(
                    torch.unsqueeze(h, -1) * transition, axis=0
                ) * to_update[i_trial - 1] + alpha[i_trial - 1] * (1 - to_update[i_trial - 1])
            # else:
            #    alpha = alpha.reshape(-1, nb_blocklengths, nb_typeblocks)
            #    alpha[i_trial, 0, 0] = 0.5
            #    alpha[i_trial, 0, -1] = 0.5
            #    alpha = alpha.reshape(-1, nb_blocklengths * nb_typeblocks)
            h = alpha[i_trial] * lks[i_trial].repeat(nb_blocklengths)
            h = h / torch.unsqueeze(torch.sum(h, axis=-1), -1)
        else:
            if i_trial > 0:
                alpha[i_trial, :] = alpha[i_trial - 1, :]

    predictive = torch.sum(alpha.reshape(-1, nb_blocklengths, nb_typeblocks), 1)
    Pis = predictive[:, 0] * gamma + predictive[:, 1] * (1 - gamma)

    return 1 - Pis


def generate_source_ids(number_of_neurons):
    combinations_neuronids = []
    for x in itertools.combinations(range(number_of_neurons), 2):
        combinations_neuronids.append([x[0], x[1]])

    combinations_neuronids = np.asarray(combinations_neuronids)
    return combinations_neuronids


def compute_information_decomposition(decoding_variable, neural_data):
    # always same region
    # neural data is in neurons x trials
    targets = decoding_variable
    sources = generate_source_ids(neural_data.shape[0])

    pid_information = np.zeros((len(sources), 4))  # neuronsC2 x 4
    coinformation_data = np.zeros((len(sources), 4))  # neuronsC2 x 4

    for idx in tqdm(range(len(sources)), desc="Running for all sources", leave=False):
        s1 = sources[idx][0]
        s2 = sources[idx][1]
        X1 = np.asarray(neural_data[s1, :], dtype=np.int32)
        X2 = np.asarray(neural_data[s2, :], dtype=np.int32)
        Y = np.asarray(targets, dtype=np.int32)
        u1, u2, red, syn = compute_pid_unbiased(Y, X1, X2)
        coinfo, mi_yx1x2, mi_yx1, mi_yx2 = coinformation(Y, X1, X2)
        pid_information[idx, :] = u1, u2, red, syn
        coinformation_data[idx, :] = mi_yx1, mi_yx2, coinfo, mi_yx1x2

    # now to organize this?
    # nah, unique information would just be the mean of the first two
    # red and syn  are fine
    # yx1 and yx2 mutual info are also similar to UI
    # the other two are trivariate

    return pid_information, coinformation_data


def computeMI(hidden_state, target, unbiased=False):
    """compute the mutual information between each neuron and a target variable

    Args:
        hidden_state (np.array): timepoints x neurons, discretized already or trials x neurons
        target (np.array): target variable, discretized already

    Returns:
        np.array: computed mutual information
    """
    mi_scores = []
    for feature_idx in range(hidden_state.shape[1]):
        # Extract a single feature
        X_feature = hidden_state[:, feature_idx]

        # Compute MI on training set
        if unbiased:
            mi_scores.append(unbiasedMI(X_feature, target))
        else:
            mi_scores.append(MI(X_feature, target))
    mi_scores = np.asarray(mi_scores)

    return mi_scores


def plot_probability_buildup(session_data, iteration):
    max_number_of_steps = 7

    last_time_steps = session_data[session_data["trial_end"] == 1]

    correct_trials = np.array(
        last_time_steps[last_time_steps["correct_action_taken"] == 1][
            "trial_within_session"
        ].values,
        dtype=int,
    )

    incorrect_trials = np.array(
        last_time_steps[last_time_steps["correct_action_taken"] == 0][
            "trial_within_session"
        ].values,
        dtype=int,
    )

    correct_trial_buildup = np.zeros((len(correct_trials), max_number_of_steps))
    incorrect_trial_buildup = np.zeros((len(incorrect_trials), max_number_of_steps))

    for idx, trial_number in enumerate(correct_trials):
        correct_trial_buildup[idx, :] = session_data[
            session_data["trial_within_session"] == trial_number
        ].correct_action_prob.values

    for idx, trial_number in enumerate(incorrect_trials):
        incorrect_trial_buildup[idx, :] = session_data[
            session_data["trial_within_session"] == trial_number
        ].correct_action_prob.values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        np.arange(max_number_of_steps),
        np.mean(correct_trial_buildup, 0),
        yerr=np.std(correct_trial_buildup, 0) / 2,
        fmt="o-",
        label="Correct Trials",
    )

    ax.errorbar(
        np.arange(max_number_of_steps),
        np.mean(incorrect_trial_buildup, 0),
        yerr=np.std(incorrect_trial_buildup, 0) / 2,
        fmt="o-",
        label="Incorrect Trials",
    )

    ax.set_ylim([0, 1.2])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Correct action probability")
    plt.legend()
    ax.axhline(y=0.9, color="r", linestyle="--", label="decision-threshold")
    ax.set_title(f"Model behavior at iteration {iteration}")
    plt.show()


def plot_probability_buildup_noaction(session_data, iteration):

    max_number_of_steps = 7

    last_time_steps = session_data[session_data["trial_end"] == 1]
    noaction_trials = np.array(
        last_time_steps[last_time_steps["action_taken"] == 0]["trial_within_session"].values,
        dtype=int,
    )

    noaction_trial_buildup = np.zeros((len(noaction_trials), max_number_of_steps))

    for idx, trial_number in enumerate(noaction_trials):
        noaction_trial_buildup[idx, :] = session_data[
            session_data["trial_within_session"] == trial_number
        ].correct_action_prob.values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        np.arange(max_number_of_steps),
        np.mean(noaction_trial_buildup, 0),
        yerr=np.std(noaction_trial_buildup, 0) / 2,
        fmt="o-",
        label="No action taken trials",
    )

    ax.set_ylim([0, 1.2])
    ax.set_xlabel("Time step")
    ax.set_ylabel("Correct action probability")
    ax.axhline(y=0.9, color="r", linestyle="--", label="decision-threshold")
    plt.legend()

    ax.set_title(f"Model behavior at iteration {iteration}")
    plt.show()


def compute_concordant_proportions(df_group):
    return np.asarray(
        [
            np.sum(df_group["concordant_trial"].values == 0),
            np.sum(df_group["concordant_trial"].values == 1),
        ]
    )


def plot_probability_buildup_per_strength(session_data, iteration, decision_threshold=0.9):

    max_number_of_steps = 7

    last_time_steps = session_data[session_data["trial_end"] == 1]

    correct_trials = np.array(
        last_time_steps[
            (last_time_steps["correct_action_taken"] == 1) & (last_time_steps["action_taken"] == 1)
        ]["trial_within_session"].values,
        dtype=int,
    )

    incorrect_trials = np.array(
        last_time_steps[
            (last_time_steps["correct_action_taken"] == 0) & (last_time_steps["action_taken"] == 1)
        ]["trial_within_session"].values,
        dtype=int,
    )

    # also add the no action trials
    noaction_trials = np.array(
        last_time_steps[last_time_steps["action_taken"] == 0]["trial_within_session"].values,
        dtype=int,
    )

    # number of trial strengths
    unique_strengths = np.sort(session_data.trial_strength.unique())
    correct_trial_buildup = np.zeros((len(unique_strengths), max_number_of_steps, 2))
    incorrect_trial_buildup = np.zeros((len(unique_strengths), max_number_of_steps, 2))
    noaction_trial_buildup = np.zeros((len(unique_strengths), max_number_of_steps, 2))

    trial_proportions = np.zeros((len(unique_strengths), 2, 3))  # correct, incorrect and no-action

    # correct trials
    for trial_strength, df_group in session_data[
        session_data["trial_within_session"].isin(correct_trials)
    ].groupby("trial_strength"):
        correct_action_buildup_mean = (
            df_group.groupby("rnn_step_index")["correct_action_prob"].mean().values
        )
        correct_action_buildup_std = (
            df_group.groupby("rnn_step_index")["correct_action_prob"].std().values
        )
        location = np.argwhere(trial_strength == unique_strengths)
        correct_trial_buildup[location, :, 0] = correct_action_buildup_mean
        correct_trial_buildup[location, :, 1] = correct_action_buildup_std
        trial_proportions[location, :, 0] = compute_concordant_proportions(df_group)

    # incorrect trials
    for trial_strength, df_group in session_data[
        session_data["trial_within_session"].isin(incorrect_trials)
    ].groupby("trial_strength"):
        incorrect_action_buildup_mean = (
            df_group.groupby("rnn_step_index")["correct_action_prob"].mean().values
        )
        incorrect_action_buildup_std = (
            df_group.groupby("rnn_step_index")["correct_action_prob"].std().values
        )
        location = np.argwhere(trial_strength == unique_strengths)
        incorrect_trial_buildup[location, :, 0] = incorrect_action_buildup_mean
        incorrect_trial_buildup[location, :, 1] = incorrect_action_buildup_std
        trial_proportions[location, :, 1] = compute_concordant_proportions(df_group)

    # no action trials
    counts = np.zeros((len(unique_strengths)))
    for trial_strength, df_group in session_data[
        session_data["trial_within_session"].isin(noaction_trials)
    ].groupby("trial_strength"):
        no_action_buildup_mean = (
            df_group.groupby("rnn_step_index")["correct_action_prob"].mean().values
        )
        no_action_buildup_std = (
            df_group.groupby("rnn_step_index")["correct_action_prob"].std().values
        )
        location = np.argwhere(trial_strength == unique_strengths)
        noaction_trial_buildup[location, :, 0] = no_action_buildup_mean
        noaction_trial_buildup[location, :, 1] = no_action_buildup_std
        counts[location] = df_group.shape[0] / max_number_of_steps
        trial_proportions[location, :, 2] = compute_concordant_proportions(df_group)

    # compute fraction of correct and incorrect trials for every stimulus strength
    fraction_corrects = np.zeros((len(unique_strengths), 2))
    for group, df in last_time_steps.groupby(["trial_strength", "correct_action_taken"]):
        strength = group[0]
        idx = int(group[1])
        fraction_index = np.argwhere(strength == unique_strengths)
        fraction_corrects[fraction_index, idx] = df.shape[0]

    # now plot
    title_fractions = np.round(fraction_corrects[:, 1] / np.sum(fraction_corrects, axis=1), 2)

    fig, axes = plt.subplots(figsize=(16, 8), ncols=3, nrows=2, sharex=True, sharey=True)

    for idx, ax in enumerate(axes.flatten()):

        trial_strength = unique_strengths[idx]
        factor = np.sum(trial_proportions[idx, :], axis=1, keepdims=True)
        ax.errorbar(
            np.arange(max_number_of_steps),
            correct_trial_buildup[idx, :, 0],
            yerr=correct_trial_buildup[idx, :, 1] / 2,
            fmt="o-",
            label="Correct Trials",
            color="blue",
            alpha=0.75,
        )
        ax.errorbar(
            np.arange(max_number_of_steps),
            incorrect_trial_buildup[idx, :, 0],
            yerr=incorrect_trial_buildup[idx, :, 1] / 2,
            fmt="o-",
            label="Incorrect Trials",
            color="red",
            alpha=0.75,
        )
        ax.errorbar(
            np.arange(max_number_of_steps),
            noaction_trial_buildup[idx, :, 0],
            yerr=noaction_trial_buildup[idx, :, 1] / 2,
            fmt="o-",
            label="No action Trials",
            color="black",
            alpha=0.75,
        )

        # plot insets for concordant vs non-concordant trials
        A = trial_proportions[idx, :] / factor

        ax_inset = inset_axes(
            ax, width="20%", height="15%", loc="upper left"
        )  # Adjust size and location
        A = A.T
        ax_inset.bar(
            np.arange(2) - 0.33, A[0, :], width=0.33, label="correct", alpha=0.75, color="blue"
        )
        ax_inset.bar(np.arange(2), A[1, :], width=0.33, label="incorrect", alpha=0.75, color="red")
        ax_inset.bar(
            np.arange(2) + 0.33, A[2, :], width=0.33, label="noaction", alpha=0.75, color="black"
        )
        ax_inset.set_xticks(np.arange(2), ["NC", "C"])
        ax_inset.set_ylim(0, 1)

        ax_inset.spines["top"].set_visible(False)
        ax_inset.spines["right"].set_visible(False)

        # if idx == 0:
        #     ax_inset.legend()
        ax.set_title(
            f"Strength={trial_strength}, Fraction correct:{title_fractions[idx]}, No action: {counts[idx]}"
        )
        ax.set_ylim([0, 1.3])
        ax.set_xlabel("Time step")
        ax.set_ylabel("Correct action probability")
        ax.axhline(y=decision_threshold, color="r", linestyle="--", label="decision-threshold")

    ax.legend()
    fig.suptitle(f"Model behavior at iteration {iteration}")
    # plt.tight_layout()
    plt.show()


def plot_probability_buildup_noaction_per_strength(session_data, iteration):

    max_number_of_steps = 7

    last_time_steps = session_data[session_data["trial_end"] == 1]
    noaction_trials = np.array(
        last_time_steps[last_time_steps["action_taken"] == 0]["trial_within_session"].values,
        dtype=int,
    )

    unique_strengths = np.sort(session_data.trial_strength.unique())
    noaction_trial_buildup = np.zeros((len(unique_strengths), max_number_of_steps, 2))

    counts = np.zeros((len(unique_strengths)))
    for trial_strength, df_group in session_data[
        session_data["trial_within_session"].isin(noaction_trials)
    ].groupby("trial_strength"):
        no_action_buildup_mean = (
            df_group.groupby("rnn_step_index")["correct_action_prob"].mean().values
        )
        no_action_buildup_std = (
            df_group.groupby("rnn_step_index")["correct_action_prob"].std().values
        )
        location = np.argwhere(trial_strength == unique_strengths)
        noaction_trial_buildup[location, :, 0] = no_action_buildup_mean
        noaction_trial_buildup[location, :, 1] = no_action_buildup_std
        counts[location] = df_group.shape[0] / max_number_of_steps

    fig, axes = plt.subplots(figsize=(12, 5), ncols=3, nrows=2, sharex=True, sharey=True)

    for idx, ax in enumerate(axes.flatten()):

        if counts[idx] == 0:
            ax.set_title(f"Strength={trial_strength}, trials = {counts[idx]}")
            continue
        trial_strength = unique_strengths[idx]
        ax.errorbar(
            np.arange(max_number_of_steps),
            noaction_trial_buildup[idx, :, 0],
            yerr=noaction_trial_buildup[idx, :, 1] / 2,
            fmt="o-",
            label="No action Trials",
        )

        ax.set_title(f"Strength={trial_strength}, trials = {counts[idx]}")
        ax.set_ylim([0, 1.2])
        ax.set_xlabel("Time step")
        ax.set_ylabel("Correct action probability")
        ax.axhline(y=0.9, color="r", linestyle="--", label="decision-threshold")

    fig.suptitle(f"Model behavior at iteration {iteration} for no action")
    plt.tight_layout()
    plt.show()


def collate_frozen_behavior(location):
    files = glob(f"{location}/*rnn*.pkl")
    numbered_files = [f for f in files if re.search(r"\d+.pkl$", f)]
    sorted_files = sorted(numbered_files, key=lambda x: int(re.search(r"(\d+).pkl$", x).group(1)))

    proportions = []
    trial_strengths = np.asarray([0.0, 0.5, 1, 1.5, 2, 2.5])
    trial_correct_values = []
    checkpoints = []

    # we want a alphabetical sort of the iterations

    for filename in sorted_files:

        checkpoint_number = (
            filename.rsplit("/")[-1].rsplit("rnn_ann_model_results_10units_")[-1].rstrip(".pkl")
        )
        checkpoints.append(int(checkpoint_number))

        trial_corrects = np.zeros((6))
        session_props = np.zeros((3))
        session_data = load_session(filename)
        groupa = (
            session_data[session_data["trial_end"] == 1]
            .groupby(["action_taken", "correct_action_taken"])["trial_index"]
            .size()
            .reset_index(name="count")
        )
        groupa["proportion"] = groupa["count"] / groupa["count"].sum()
        groupa["correct_action_taken"] = groupa["correct_action_taken"].astype("int")
        groups = (
            session_data[
                (session_data["trial_end"] == 1) & session_data["correct_action_taken"] == 1
            ]
            .groupby("trial_strength")
            .size()
        )
        for idy in range(len(groupa["proportion"].values)):
            session_props[idy] = groupa["proportion"].values[idy]
        proportions.append(session_props)
        for v in groups.index.values:
            idx = np.argwhere(v == trial_strengths).reshape(
                -1,
            )[0]
            trial_corrects[idx] = groups.loc[v]
        trial_correct_values.append(trial_corrects)

    return np.asarray(proportions), np.asarray(trial_correct_values), np.asarray(checkpoints)


def plot_frozen_behavior(proportions, trial_correct_values, checkpoint_numbers, frozen=True):
    fig, ax = plt.subplots(figsize=(8, 4), ncols=2)
    ax[0].plot(proportions[:, 0], marker=".", label="No response")
    ax[0].plot(proportions[:, 1], marker=".", label="Incorrect response")
    ax[0].plot(proportions[:, 2], marker=".", label="Correct response")
    ax[0].set_ylim(0, 1.2)
    ax[0].set_ylabel("Proportion")
    ax[0].legend()
    ax[0].set_xticks(np.arange(len(checkpoint_numbers)), checkpoint_numbers, rotation=90)

    strength = [0, 0.5, 1, 1.5, 2, 2.5]
    for idx in range(6):
        ax[1].plot(trial_correct_values[:, idx], marker=".", label=f"{strength[idx]}")
    ax[1].legend()
    # ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("Number of trials")
    ax[1].legend()
    if frozen:
        ax[0].set_xlabel("Iterations after rnn-freeze")
        ax[1].set_xlabel("Iterations after rnn-freeze")
    else:
        ax[0].set_xlabel("Iterations")
        ax[1].set_xlabel("Iterations")

    ax[1].set_xticks(np.arange(len(checkpoint_numbers)), checkpoint_numbers, rotation=90)
    ax[1].set_title("Strength of trials given correct response")
    ax[0].set_title("Proportion of responses")
    plt.tight_layout()


def plot_mutual_info_heatmaps(
    trial_side_mi,
    block_side_mi,
    action_side_mi,
    bayes_prior_mi,
    correct_action_mi,
    iteration_number,
):
    fig, ax = plt.subplots(figsize=(4 * 5, 4), ncols=4, sharex=True, sharey=True)

    sns.heatmap(trial_side_mi.T, ax=ax[0], cmap="viridis", linecolor="k", linewidths=0.25)
    sns.heatmap(block_side_mi.T, ax=ax[1], cmap="viridis", linecolor="k", linewidths=0.25)
    sns.heatmap(action_side_mi.T, ax=ax[2], cmap="viridis", linecolor="k", linewidths=0.25)
    sns.heatmap(bayes_prior_mi.T, ax=ax[3], cmap="viridis", linecolor="k", linewidths=0.25)
    # sns.heatmap(correct_action_mi.T, ax=ax[4], cmap="viridis", linecolor="k", linewidths=0.25)

    ax[-1].set_xticklabels(labels=["t0", "t1", "t2", "t3", "t4", "t5", "t6"])
    for idx in range(4):
        ax[idx].set_xlabel("Time step")

    ax[0].set_ylabel("Hidden neuron id")

    ax[0].set_title("Stimulus side")
    ax[1].set_title("Block side")
    ax[2].set_title("Action side")
    ax[3].set_title("Bayes prior")
    # ax[4].set_title("Correct action probability")
    plt.suptitle(f"Itx : {iteration_number}")
    plt.tight_layout()


def compute_mutual_information(session_data, unbiased=False):
    # first normalize the entire array
    tend = session_data[session_data["rnn_step_index"] == 6]
    side = tend["trial_side"].values
    action = tend["action_side"].values
    action[np.isnan(action)] = 0
    block_side = tend["block_side"].values
    bayesian_prior = optimal_bayesian(side, 0.02).detach().numpy()
    bayes_discrete = probability_binning(bayesian_prior)

    trial_side_mi = np.zeros((7, 10))  # 7 steps, 10 neurons, mean and std
    block_side_mi = np.zeros((7, 10))
    action_side_mi = np.zeros((7, 10))
    left_stim_mi = np.zeros((7, 10))
    right_stim_mi = np.zeros((7, 10))
    bayes_prior_mi = np.zeros((7, 10))
    correct_action_mi = np.zeros((7, 10))

    all_hidden_state = np.concatenate(session_data.hidden_state)
    discrete_hidden_state = hidden_layer_binning(all_hidden_state)

    for x in range(0, 7):
        t_x = session_data.loc[
            session_data["rnn_step_index"] == x,
            [
                "block_side",
                "trial_within_session",
                "trial_strength",
                "action_side",
                "trial_side",
                "correct_action_taken",
                "hidden_state",
                "left_stimulus",
                "right_stimulus",
                "rnn_step_index",
                "correct_action_prob",
                "left_action_prob",
                "right_action_prob",
                "concordant_trial",
            ],
        ]

        indexes = t_x.index.values
        hidden_state_t_x = discrete_hidden_state[indexes, :]

        # now what
        results_trial_side_x = computeMI(hidden_state_t_x, t_x["trial_side"].values, unbiased)
        results_block_side_x = computeMI(hidden_state_t_x, t_x["block_side"].values, unbiased)

        left_stim_discrete = equipopulated_binning(t_x["left_stimulus"].values)
        right_stim_discrete = equipopulated_binning(t_x["right_stimulus"].values)
        correct_action_discrete = probability_binning(t_x["correct_action_prob"].values)

        results_left_stimulus_x = computeMI(hidden_state_t_x, left_stim_discrete, unbiased)
        results_right_stimulus_x = computeMI(hidden_state_t_x, right_stim_discrete, unbiased)
        results_correct_action_x = computeMI(hidden_state_t_x, correct_action_discrete, unbiased)

        # now what
        # action, block and bayes
        actions = session_data.loc[
            session_data["rnn_step_index"] == 6, ["action_side"]
        ].values.reshape(
            -1,
        )
        valid_mask = ~np.isnan(actions)

        # get locations
        # nan_indices = np.where(nan_mask)[0]
        # let's flip this
        # set nans to 0
        actions[~valid_mask] = 0
        results_action_side_x = computeMI(hidden_state_t_x, actions, unbiased)

        # hidden_state_nonan = hidden_state_t_x[valid_mask, :]
        # actions = actions[valid_mask]

        # results_action_side_x = computeMI(hidden_state_nonan, actions, unbiased)

        # dsc = t_x["correct_action_prob"] > 0.5
        # dsc = dsc.values
        # dsc = np.asarray(dsc, dtype="int")

        # replaced this with block prior
        results_bayes_prior_x = computeMI(hidden_state_t_x, bayes_discrete, unbiased)

        # now compute mutual information for all states, maybe actions when

        trial_side_mi[x, :] = results_trial_side_x
        block_side_mi[x, :] = results_block_side_x
        action_side_mi[x, :] = results_action_side_x
        left_stim_mi[x, :] = results_left_stimulus_x
        right_stim_mi[x, :] = results_right_stimulus_x
        bayes_prior_mi[x, :] = results_bayes_prior_x
        correct_action_mi[x, :] = results_correct_action_x

    return (
        trial_side_mi,
        block_side_mi,
        action_side_mi,
        left_stim_mi,
        right_stim_mi,
        bayes_prior_mi,
        correct_action_mi,
    )


def organize_data_classification(session_data, timestep):
    """organize the data for the     linear classification

    Args:
        session_data (pd.df): dataframe with rnn behavior
        timestep (int): timestep for session_data
    """
    t_x = session_data.loc[
        session_data["rnn_step_index"] == timestep,
        [
            "block_side",
            "trial_within_session",
            "trial_strength",
            "action_side",
            "trial_side",
            "correct_action_taken",
            "hidden_state",
            "left_stimulus",
            "right_stimulus",
            "rnn_step_index",
            "correct_action_prob",
            "left_action_prob",
            "right_action_prob",
            "concordant_trial",
        ],
    ]

    # get block and action side
    block_side = t_x["block_side"].values
    stimulus_side = t_x["trial_side"].values

    all_hidden_state = np.concatenate(session_data.hidden_state)
    discrete_hidden_state = hidden_layer_binning(all_hidden_state)

    indexes = t_x.index.values
    hidden_state_t_x_discrete = discrete_hidden_state[indexes, :]
    hidden_state_t_x_normal = all_hidden_state[indexes, :]

    return block_side, stimulus_side, hidden_state_t_x_discrete, hidden_state_t_x_normal


def parallelize_computation(session_data, t_x, hidden_state, bayes_discrete):

    indexes = t_x.index.values
    hidden_state_t_x = hidden_state[
        indexes, :
    ].T  # because informationd decomposition expects neurons x trials

    pid_information_trial, coinformation_data_trial = compute_information_decomposition(
        t_x["trial_side"].values,
        hidden_state_t_x,
    )

    pid_information_block, coinformation_data_block = compute_information_decomposition(
        t_x["block_side"].values,
        hidden_state_t_x,
    )

    actions = session_data.loc[
        session_data["rnn_step_index"] == 6, ["action_side"]
    ].values.reshape(
        -1,
    )
    valid_mask = ~np.isnan(actions)

    actions[~valid_mask] = 0
    pid_information_action_side, coinformation_data_action_side = (
        compute_information_decomposition(
            actions,
            hidden_state_t_x,
        )
    )

    # replaced this with block prior
    pid_information_bayes, coinformation_data_bayes = compute_information_decomposition(
        bayes_discrete,
        hidden_state_t_x,
    )

    # now stack all pid informations

    info_decom_trial = np.hstack([pid_information_trial, coinformation_data_trial])
    info_decom_block = np.hstack([pid_information_block, coinformation_data_block])
    info_decom_action = np.hstack([pid_information_action_side, coinformation_data_action_side])
    info_decom_bayes = np.hstack([pid_information_bayes, coinformation_data_bayes])

    return info_decom_trial, info_decom_block, info_decom_action, info_decom_bayes


def information_decomposition_all(session_data):
    """compute information decomposition across hidden units, fore each epoch

    Args:
        session_data (pd.df): Session behavior
    """
    # calcualte bayes prior and discretize it
    tend = session_data[session_data["rnn_step_index"] == 6]
    side = tend["trial_side"].values
    action = tend["action_side"].values
    action[np.isnan(action)] = 0
    bayesian_prior = optimal_bayesian(side, 0.02).detach().numpy()
    bayes_discrete = probability_binning(bayesian_prior)

    # # compute pids, stack and then append : makes sense
    # trial_side_pid = []
    # block_side_pid = []
    # action_side_pid = []
    # bayes_prior_pid = []

    all_hidden_state = np.concatenate(session_data.hidden_state)
    discrete_hidden_state = hidden_layer_binning(all_hidden_state)

    t_x_splits = []

    for x in range(0, 7):
        t_x = session_data.loc[
            session_data["rnn_step_index"] == x,
            [
                "block_side",
                "trial_within_session",
                "trial_strength",
                "action_side",
                "trial_side",
                "correct_action_taken",
                "hidden_state",
                "left_stimulus",
                "right_stimulus",
                "rnn_step_index",
                "correct_action_prob",
                "left_action_prob",
                "right_action_prob",
                "concordant_trial",
            ],
        ]
        t_x_splits.append(t_x)

    # now what
    results = Parallel(n_jobs=-1)(
        delayed(parallelize_computation)(
            session_data, t_x_splits[idx], discrete_hidden_state, bayes_discrete
        )
        for idx in tqdm(range(len(t_x_splits)), desc="Running for different tx")
    )

    trial_side_pid, block_side_pid, action_side_pid, bayes_prior_pid = zip(*results)

    final_trial_side_pid = np.stack(trial_side_pid, axis=0)
    final_block_side_pid = np.stack(block_side_pid, axis=0)
    final_action_side_pid = np.stack(action_side_pid, axis=0)
    final_bayes_prior_pid = np.stack(bayes_prior_pid, axis=0)

    return final_trial_side_pid, final_block_side_pid, final_action_side_pid, final_bayes_prior_pid
