from one.api import ONE
from brainbox.population.decode import get_spike_counts_in_bins
import numpy as np
import pandas as pd
from ibl_info.utility import (
    aggregated_regions_time_intervals,
    discretize_neural_data,
    maintain_neural_count,
    subsample,
)
from iblatlas.atlas import BrainRegions
import matplotlib.pyplot as plt
import seaborn as sns


def gather_data_stim(trials_df, spikes_probe0, spikes_probe1, time_window=[0, 0.1]):
    """
    Gather data around stimulus onset from multiple probes

    Args:
        trials_df (pandas.df): Trial dataframe
        spikes_probe0 (dict): Neural data from probe
        spikes_probe1 (dict): Neural data from probe
        time_window (list, optional): Time window in which to pool the data. Defaults to [0,0.1].

    Returns:
        spike_count_stim_probe0 (np.array): Spike counts for probe 0, neurons x trials
        spike_count_stim_probe1 (np.array): Spike counts for probe 1, neurons x trials
        cluster_id_probe0 (np.array) : Neural cluster ids for probe 0
        cluster_id_probe1 (np.array) : Neural cluster ids for probe 1
        decoding_variable (np.array) : Stimulus side
    """
    correct_trials = trials_df[trials_df["feedbackType"] == 1]

    left_stim_trials = correct_trials[correct_trials.contrastLeft > 0]
    right_stim_trials = correct_trials[correct_trials.contrastRight > 0]

    stimon_times = np.concatenate(
        [left_stim_trials.stimOn_times.values, right_stim_trials.stimOn_times.values]
    )
    decoding_variable = np.concatenate(
        [np.ones((left_stim_trials.shape[0])), 0 * np.ones((right_stim_trials.shape[0]))]
    )
    events_stim_tw = np.array([stimon_times + time_window[0], stimon_times + time_window[1]]).T

    # Neurons x Trials
    spike_count_stim_probe0, cluster_id_probe0 = get_spike_counts_in_bins(
        spikes_probe0["times"], spikes_probe0["clusters"], events_stim_tw
    )
    spike_count_stim_probe1, cluster_id_probe1 = get_spike_counts_in_bins(
        spikes_probe1["times"], spikes_probe1["clusters"], events_stim_tw
    )

    # for now, we don't use the single-cell peths

    return (
        spike_count_stim_probe0,
        spike_count_stim_probe1,
        cluster_id_probe0,
        cluster_id_probe1,
        decoding_variable,
    )


def gather_data_choice(trials_df, spikes_probe0, spikes_probe1, time_window=[-0.2, 0.0]):
    """
    Gather data around choice interval from multiple probes

    Args:
        trials_df (pandas.df): Trial dataframe
        spikes_probe0 (dict): Neural data from probe
        spikes_probe1 (dict): Neural data from probe
        time_window (list, optional): Time window in which to pool the data. Defaults to [0,0.1].

    Returns:
        spike_count_choice_probe0 (np.array): Spike counts for probe 0, neurons x trials
        spike_count_choice_probe1 (np.array): Spike counts for probe 1, neurons x trials
        cluster_id_probe0 (np.array) : Neural cluster ids for probe 0
        cluster_id_probe1 (np.array) : Neural cluster ids for probe 1
        decoding_variable (np.array) : Choice direction
    """

    left_choice_trials = trials_df[trials_df.choice == 1]
    right_choice_trials = trials_df[trials_df.choice == -1]

    choice_times = np.concatenate(
        [
            left_choice_trials.firstMovement_times.values,
            right_choice_trials.firstMovement_times.values,
        ]
    )
    decoding_variable = np.concatenate(
        [np.ones((left_choice_trials.shape[0])), 0 * np.ones((right_choice_trials.shape[0]))]
    )
    events_choice_tw = np.array([choice_times + time_window[0], choice_times + time_window[1]]).T

    # Neurons x Trials
    spike_count_choice_probe0, cluster_id_probe0 = get_spike_counts_in_bins(
        spikes_probe0["times"], spikes_probe0["clusters"], events_choice_tw
    )
    spike_count_choice_probe1, cluster_id_probe1 = get_spike_counts_in_bins(
        spikes_probe1["times"], spikes_probe1["clusters"], events_choice_tw
    )

    # for now, we don't use the single-cell peths

    return (
        spike_count_choice_probe0,
        spike_count_choice_probe1,
        cluster_id_probe0,
        cluster_id_probe1,
        decoding_variable,
    )


def gather_data_feedback(trials_df, spikes_probe0, spikes_probe1, time_window=[0.0, 0.2]):
    """
    Gather data around feedback interval from multiple probes

    Args:
        trials_df (pandas.df): Trial dataframe
        spikes_probe0 (dict): Neural data from probe
        spikes_probe1 (dict): Neural data from probe
        time_window (list, optional): Time window in which to pool the data. Defaults to [0,0.1].

    Returns:
        spike_count_feedback_probe0 (np.array): Spike counts for probe 0, neurons x trials
        spike_count_feedback_probe1 (np.array): Spike counts for probe 1, neurons x trials
        cluster_id_probe0 (np.array) : Neural cluster ids for probe 0
        cluster_id_probe1 (np.array) : Neural cluster ids for probe 1
        decoding_variable (np.array) : Feedback valence
    """

    correct_feedback_trials = trials_df[trials_df.feedbackType == 1]
    incorrect_feedback_trials = trials_df[trials_df.feedbackType == -1]

    feedback_times = np.concatenate(
        [
            correct_feedback_trials.feedback_times.values,
            incorrect_feedback_trials.feedback_times.values,
        ]
    )
    decoding_variable = np.concatenate(
        [
            np.ones((correct_feedback_trials.shape[0])),
            0 * np.ones((incorrect_feedback_trials.shape[0])),
        ]
    )
    events_feedback_tw = np.array(
        [feedback_times + time_window[0], feedback_times + time_window[1]]
    ).T

    # Neurons x Trials
    spike_count_feedback_probe0, cluster_id_probe0 = get_spike_counts_in_bins(
        spikes_probe0["times"], spikes_probe0["clusters"], events_feedback_tw
    )
    spike_count_feedback_probe1, cluster_id_probe1 = get_spike_counts_in_bins(
        spikes_probe1["times"], spikes_probe1["clusters"], events_feedback_tw
    )

    # for now, we don't use the single-cell peths

    return (
        spike_count_feedback_probe0,
        spike_count_feedback_probe1,
        cluster_id_probe0,
        cluster_id_probe1,
        decoding_variable,
    )


def gather_data_prior(
    trials_df, spikes_probe0, spikes_probe1, time_window=[-0.6, -0.1], prior="action-kernel"
):

    # prior can be glm-hmm or action kernel
    if prior == "action-kernel":
        decoding_variable = trials_df["prior-binary"]
    elif prior == "glm-hmm":
        decoding_variable = trials_df["state"]

    stimonset = trials_df.stimOn_times.values
    events_prior_tw = np.array([stimonset + time_window[0], stimonset + time_window[1]]).T

    spike_count_feedback_probe0, cluster_id_probe0 = get_spike_counts_in_bins(
        spikes_probe0["times"], spikes_probe0["clusters"], events_prior_tw
    )
    spike_count_feedback_probe1, cluster_id_probe1 = get_spike_counts_in_bins(
        spikes_probe1["times"], spikes_probe1["clusters"], events_prior_tw
    )

    return (
        spike_count_feedback_probe0,
        spike_count_feedback_probe1,
        cluster_id_probe0,
        cluster_id_probe1,
        decoding_variable,
    )


def cleanup_data(neural_data, regions):
    """
    Throw away rows with root, void and other undesirable regions

    Args:
        neural_data (np.array): neurons x trials
        regions (np.array): acronym for neurons

    Returns:
        neural_data (np.array) : neurons x trials, cleaned
        regions (np.array) : acronym for cleaned up neurons

    """

    # if region not in region_info.csv, throw away
    throw_away = ["root", "void"]
    bad_indices = np.isin(regions, throw_away) == True
    print(f"Neurons thrown away: {regions[bad_indices]}")

    # TODO: maybe throw away regions that don't have enough neurons

    neural_data = neural_data[~bad_indices, :]
    regions = regions[~bad_indices]

    return np.asarray(neural_data, dtype=np.int32), regions


def combine_probes(
    spike_count_stim_probe0,
    spike_count_stim_probe1,
    regions_probe0,
    regions_probe1,
    aggregate=False,
    average=False,
    discretize=False,
    method="all",
):
    """
    Combine probes from same session

    Args:
        spike_count_stim_probe0 (dict): Spike data from probe 0
        spike_count_stim_probe1 (dict): Spike data from probe 1
        regions_probe0 (np.array): Regional acronyms from probe 0
        regions_probe1 (np.array): Regional acronyms from probe 1
        aggregate (bool, optional): Aggreate regions or not. Defaults to False.

    Returns:
        neural_data (np.array) neurons x trials
        regions (np.array): neuron regions
    """

    # best place to run filter to see if there are enough neurons from each region
    MINIMUM_NUMBER = 10

    if aggregate:
        # combine neurons from multiple regions into one big chunk
        # spike_count_probe is neurons x trials
        aggregate_neural_data = np.vstack([spike_count_stim_probe0, spike_count_stim_probe1])
        aggregate_regions = np.concatenate([regions_probe0, regions_probe1])

        # now run aggregation
        aggregate_neural_data, aggregate_regions = aggregated_regions_time_intervals(
            aggregate_neural_data.T, aggregate_regions, average=average
        )  # because the function expects trials x neurons
        # now we clean up the data?
        neural_data, regions = cleanup_data(aggregate_neural_data.T, aggregate_regions)
    else:
        neural_data_probe0, regions_probe0 = cleanup_data(spike_count_stim_probe0, regions_probe0)
        neural_data_probe1, regions_probe1 = cleanup_data(spike_count_stim_probe1, regions_probe1)

        # concatenate regions and neural data
        neural_data = np.vstack([neural_data_probe0, neural_data_probe1])
        regions = np.concatenate([regions_probe0, regions_probe1])

        ## get regions and count
        neural_data, regions = maintain_neural_count(neural_data, regions)

    # plot_neurons(neural_data, regions)
    return neural_data, regions


# gather data from single or merged probes, same as functions before
# just rewritten


def prepare_ephys_data(spikes, clusters, intervals, regions, minimum_units=10):

    # find valid regions

    brainreg = BrainRegions()
    beryl_regions = brainreg.acronym2acronym(clusters["acronym"], mapping="Beryl")

    if len(regions) == 0:
        regions = beryl_regions

    # print found regions
    for region in regions:
        # find all clusters in region (where region can be a list of regions)
        region_mask = np.isin(beryl_regions, region)
        if sum(region_mask) < minimum_units:
            continue
        else:
            print(f"Region found {region}, {np.sum(region_mask)}")

    # then get data from region mask

    binned_spikes = []
    actual_regions = []
    n_units = []
    cluster_uuids_list = []
    for region in regions:
        # find all clusters in region (where region can be a list of regions)
        region_mask = np.isin(beryl_regions, region)
        if sum(region_mask) < minimum_units:
            # print(f"{(region)} below min units threshold ({minimum_units})")
            continue
        else:
            # find all spikes in those clusters
            spike_mask = np.isin(spikes["clusters"], clusters[region_mask].index)
            times_masked = spikes["times"][spike_mask]
            clusters_masked = spikes["clusters"][spike_mask]
            # record cluster uuids
            idxs_used = np.unique(clusters_masked)
            clusters_uuids = list(clusters.iloc[idxs_used]["uuids"])
            # bin spikes from those clusters

            binned, _ = get_spike_counts_in_bins(
                spike_times=times_masked, spike_clusters=clusters_masked, intervals=intervals
            )
            binned = binned.T
            binned_spikes.append(binned)
            actual_regions.append(region)
            n_units.append(sum(region_mask))
            cluster_uuids_list.append(clusters_uuids)
    return binned_spikes, actual_regions, n_units, cluster_uuids_list


def get_window(decoding_interval):

    if decoding_interval == "stim":
        time_window = [0, 0.2]
    elif decoding_interval == "choice":
        time_window = [-0.2, 0]
    elif decoding_interval == "feedback":
        time_window = [0, 0.2]
    elif decoding_interval == "action-kernel" or decoding_interval == "glm-hmm":
        time_window = [-0.6, -0.1]
    else:
        time_window = [0, 0]

    return time_window


def compute_intervals(trials_df, decoding_interval):

    time_window = get_window(decoding_interval)

    if decoding_interval == "stim":
        # get left and right visible trials
        correct_trials = trials_df[trials_df["feedbackType"] == 1]

        left_stim_trials = correct_trials[correct_trials.contrastLeft > 0]
        right_stim_trials = correct_trials[correct_trials.contrastRight > 0]

        stimon_times = np.concatenate(
            [left_stim_trials.stimOn_times.values, right_stim_trials.stimOn_times.values]
        )
        decoding_variable = np.concatenate(
            [np.ones((left_stim_trials.shape[0])), 0 * np.ones((right_stim_trials.shape[0]))]
        )
        intervals = np.array([stimon_times + time_window[0], stimon_times + time_window[1]]).T

    elif decoding_interval == "choice":
        left_choice_trials = trials_df[trials_df.choice == 1]
        right_choice_trials = trials_df[trials_df.choice == -1]

        choice_times = np.concatenate(
            [
                left_choice_trials.firstMovement_times.values,
                right_choice_trials.firstMovement_times.values,
            ]
        )
        decoding_variable = np.concatenate(
            [np.ones((left_choice_trials.shape[0])), 0 * np.ones((right_choice_trials.shape[0]))]
        )
        intervals = np.array([choice_times + time_window[0], choice_times + time_window[1]]).T

    elif decoding_interval == "feedback":

        correct_feedback_trials = trials_df[trials_df.feedbackType == 1]
        incorrect_feedback_trials = trials_df[trials_df.feedbackType == -1]

        feedback_times = np.concatenate(
            [
                correct_feedback_trials.feedback_times.values,
                incorrect_feedback_trials.feedback_times.values,
            ]
        )
        decoding_variable = np.concatenate(
            [
                np.ones((correct_feedback_trials.shape[0])),
                0 * np.ones((incorrect_feedback_trials.shape[0])),
            ]
        )
        intervals = np.array([feedback_times + time_window[0], feedback_times + time_window[1]]).T

    elif decoding_interval == "action-kernel" or decoding_interval == "glm-hmm":
        if decoding_interval == "action-kernel":
            decoding_variable = trials_df["prior-binary"]
        elif decoding_interval == "glm-hmm":
            decoding_variable = trials_df["state"]

        stimonset = trials_df.stimOn_times.values
        intervals = np.array([stimonset + time_window[0], stimonset + time_window[1]]).T

    else:
        return NotImplementedError

    return intervals, decoding_variable


def get_congruent_incongruent_intervals(trials_df, decoding_interval):

    time_window = get_window(decoding_interval)
    all_contrasts = [1, 0.5, 0.25, 0.125, 0.0625, 0]
    low_contrasts = [0.0625, 0.125, 0.25]  # 0 is essentially random yeah.

    # let's assume all is stimulus interval for now

    left_stim_trials = trials_df[trials_df.contrastLeft >= 0]
    right_stim_trials = trials_df[trials_df.contrastRight >= 0]

    congruent_left = trials_df[(trials_df.contrastLeft >= 0) & (trials_df.probabilityLeft == 0.8)]
    congruent_right = trials_df[
        (trials_df.contrastRight >= 0) & (trials_df.probabilityLeft == 0.2)
    ]
    incongruent_left = trials_df[
        (trials_df.contrastLeft >= 0) & (trials_df.probabilityLeft == 0.2)
    ]
    incongruent_right = trials_df[
        (trials_df.contrastRight >= 0) & (trials_df.probabilityLeft == 0.8)
    ]

    congruent = pd.concat([congruent_left, congruent_right])
    incongruent = pd.concat([incongruent_left, incongruent_right])

    middling_incongruent = incongruent[
        (incongruent.contrastLeft.isin(low_contrasts))
        | (incongruent.contrastRight.isin(low_contrasts))
    ]

    # now to get all the intervals

    # for all trials
    stimon_times_all = np.concatenate(
        [left_stim_trials.stimOn_times.values, right_stim_trials.stimOn_times.values]
    )
    decoding_variable_all = np.concatenate(
        [np.ones((left_stim_trials.shape[0])), 0 * np.ones((right_stim_trials.shape[0]))]
    )

    # for congruent trials

    stimon_times_congruent = congruent.stimOn_times.values
    decoding_variable_congruent = np.concatenate(
        [np.ones((congruent_left.shape[0])), 0 * np.ones((congruent_right.shape[0]))]
    )

    # for incongruent trials

    stimon_times_incongruent = incongruent.stimOn_times.values
    decoding_variable_incongruent = np.concatenate(
        [np.ones((incongruent_left.shape[0])), 0 * np.ones((incongruent_right.shape[0]))]
    )

    # for middling incongruent trials
    stimon_times_middling = middling_incongruent.stimOn_times.values
    decoding_variable_middling = np.concatenate(
        [
            np.ones(np.sum(middling_incongruent.probabilityLeft == 0.2)),
            0 * np.ones(np.sum(middling_incongruent.probabilityLeft == 0.8)),
        ]
    )

    intervals_all = np.array(
        [stimon_times_all + time_window[0], stimon_times_all + time_window[1]]
    ).T

    intervals_congruent = np.array(
        [stimon_times_congruent + time_window[0], stimon_times_congruent + time_window[1]]  # type: ignore
    ).T

    intervals_incongruent = np.array(
        [stimon_times_incongruent + time_window[0], stimon_times_incongruent + time_window[1]]  # type: ignore
    ).T

    intervals_middling = np.array(
        [stimon_times_middling + time_window[0], stimon_times_middling + time_window[1]]  # type: ignore
    ).T

    return (
        [intervals_all, intervals_congruent, intervals_incongruent, intervals_middling],
        [
            decoding_variable_all,
            decoding_variable_congruent,
            decoding_variable_incongruent,
            decoding_variable_middling,
        ],
    )


def cleaned_regions_single_region(
    region_data, percent_of_no_spikes_threshold=0.2, firing_rate_threshold=0, plot=False
):

    # we are not using the firing rate threshold at the moment
    # otherwise we can throw away neurons that fall below a certain percentage of the mean firing rate
    # code not here
    # we pass in regional data here anyways
    # differs from old function signature
    array_no_nans = region_data[~np.isnan(region_data).any(axis=1)]
    array_no_zeros = array_no_nans[~np.all(array_no_nans == 0, axis=1)]
    num_zeros = np.sum(array_no_zeros == 0, axis=1) / array_no_zeros.shape[1]
    keep_neurons = num_zeros <= 1 - percent_of_no_spikes_threshold
    array_filtered = array_no_zeros[keep_neurons]

    thrown_away_neurons = np.where(~keep_neurons)[0]

    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))

        sns.heatmap(region_data, ax=ax, cmap="Greys")
        ax.set_title("cleaned")
        yticklabels = ax.get_yticklabels()
        for i, label in enumerate(yticklabels):
            if i in thrown_away_neurons:
                label.set_color("red")
        y_coordinates = np.arange(region_data.shape[0] + 1)

        xmin, xmax = ax.get_xlim()

        ax.hlines(
            y=y_coordinates,
            xmin=xmin,
            xmax=xmax,
            colors="black",
            linestyles="solid",
            lw=1,
        )

    return array_filtered
