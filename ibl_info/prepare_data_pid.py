import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainbox.population.decode import get_spike_counts_in_bins
from iblatlas.atlas import BrainRegions
from one.api import ONE


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
        # add another mask here to check for single units
        # i think this makes sense; pass it in
        # NOTE ::: add it here

        if sum(region_mask) < minimum_units:
            print(f"{(region)} below min units threshold ({minimum_units})")
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
        time_window = [0, 0.1]
    elif decoding_interval == "choice":
        time_window = [-0.2, 0]
    elif decoding_interval == "feedback":
        time_window = [0, 0.2]
    elif decoding_interval == "action-kernel" or decoding_interval == "glm-hmm":
        time_window = [-0.6, -0.1]
    else:
        time_window = [0, 0]

    return time_window


def get_contrast_intervals(trials_df, decoding_interval="stim"):

    time_window = get_window(decoding_interval)
    contrasts = [1, 0.25, 0.125, 0.0625, 0]
    contrast_interval = []

    for contrast in contrasts:
        trials = trials_df[
            (trials_df.contrastLeft == contrast) | (trials_df.contrastRight == contrast)
        ]
        stimon_times = trials.stimOn_times.values
        intervals = np.array([stimon_times + time_window[0], stimon_times + time_window[1]]).T
        contrast_interval.append(intervals)

    return contrast_interval


def get_new_cinc_intervals(trials_df, decoding_interval):
    """Returns intervals, stim sides and flags for congruent and incongruent trials



    Args:
        trials_df (pandas.df): Trial dataframe from ibl
        decoding_interval (string): epoch name

    Returns:
        Intervals, stim_side, congruent_flags, incongruent_flags: Neuron intervals, Stimulus Side and Flags
    """

    time_window = get_window(decoding_interval)

    left_stim = trials_df.contrastLeft >= 0
    right_stim = trials_df.contrastRight >= 0

    left_block = trials_df.probabilityLeft == 0.8
    right_block = trials_df.probabilityLeft == 0.2

    congruent_left = left_stim & left_block
    congruent_right = right_stim & right_block

    incongruent_left = left_stim & right_block
    incongruent_right = right_stim & left_block

    stimon_times = trials_df.stimOn_times.values

    intervals = np.array([stimon_times + time_window[0], stimon_times + time_window[1]]).T

    # decoding varialbe i.e, stim-side
    stim_side = []
    for idx in range(len(trials_df)):
        if left_stim.iloc[idx]:
            stim_side.append(1)
        elif right_stim.iloc[idx]:
            stim_side.append(0)
    stim_side = np.asarray(stim_side)

    congruency_flags = congruent_left | congruent_right
    incongruency_flags = incongruent_left | incongruent_right

    return intervals, stim_side, congruency_flags, incongruency_flags


def get_congruent_incongruent_intervals(trials_df, decoding_interval):

    time_window = get_window(decoding_interval)
    all_contrasts = [1, 0.25, 0.125, 0.0625, 0]
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

    # NOTE: removed the middling ones
    return (
        [intervals_all, intervals_congruent, intervals_incongruent],
        [
            decoding_variable_all,
            decoding_variable_congruent,
            decoding_variable_incongruent,
        ],
    )


def cleaned_regions_flags(
    region_data, percent_of_no_spikes_threshold=0.4, firing_rate_threshold=1.0, window=[0, 0.1]
):
    # fix the flag
    # neurons x trials

    nan_flag = np.isnan(region_data).any(axis=1)
    spike_rate = region_data / (window[1] - window[0])
    mean_spike_rate = np.mean(spike_rate, axis=1)
    keep_flag = mean_spike_rate >= firing_rate_threshold  # set as 25th percentile of global mean

    num_zeros = np.nansum(region_data == 0, axis=1) / region_data.shape[1]
    keep_neurons = num_zeros <= 1 - percent_of_no_spikes_threshold
    keep_flag = keep_flag & keep_neurons & ~nan_flag

    return keep_flag


def old_cleaner(
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
