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
            continue
        else:
            # find all spikes in those clusters
            spike_mask = np.isin(spikes["clusters"], clusters[region_mask].index)
            times_masked = spikes["times"][spike_mask]
            clusters_masked = spikes["clusters"][spike_mask]
            # record cluster uuids
            idxs_used = np.unique(clusters_masked)
            clusters_uuids = list(
                clusters.iloc[idxs_used]["cluster_id"]
            )  # note: changed cluster uuids to cluster id so that we can segment
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
        time_window = [-0.1, 0]
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

    # decoding variable i.e, stim-side
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


def get_new_cinc_intervals_choice(trials_df, decoding_interval="choice"):

    time_window = get_window(decoding_interval)

    left_stim = trials_df.contrastLeft >= 0
    right_stim = trials_df.contrastRight >= 0

    left_block = trials_df.probabilityLeft == 0.8
    right_block = trials_df.probabilityLeft == 0.2

    congruent_left = left_stim & left_block
    congruent_right = right_stim & right_block

    incongruent_left = left_stim & right_block
    incongruent_right = right_stim & left_block

    choice_times = trials_df["firstMovement_times"].values

    intervals = np.array([choice_times + time_window[0], choice_times + time_window[1]]).T

    # decoding variable, i/e, choice

    choice_side = []
    for idx in range(len(trials_df)):
        if trials_df.iloc[idx]["choice"] == -1:
            choice_side.append(0)
        elif trials_df.iloc[idx]["choice"] == 1:
            choice_side.append(1)

    choice_side = np.asarray(choice_side)

    congruency_flags = congruent_left | congruent_right
    incongruency_flags = incongruent_left | incongruent_right

    return intervals, choice_side, congruency_flags, incongruency_flags


def return_significant_cells(eid, epoch, cluster_ids):

    if epoch == "stim":
        df_epoch = pd.read_csv("./data/external/bwm_single_cell_stim_2024_10_21.csv")
    elif epoch == "choice":
        df_epoch = pd.read_csv("./data/external/bwm_single_cell_choice_2024_10_21.csv")
    else:
        raise NotImplementedError

    compact_df_stim = df_epoch[df_epoch.eid == eid]
    truncated_df = compact_df_stim[compact_df_stim["cluster_id"].isin(cluster_ids[0])]

    boolean_array = np.zeros(len(cluster_ids[0]), dtype=bool)

    for idx in range(len(cluster_ids[0])):

        xdf = truncated_df[truncated_df["cluster_id"] == cluster_ids[0][idx]]

        if len(xdf) == 0:
            boolean_array[idx] = False
        else:
            if epoch == "stim":
                xdf_pval = xdf.p_value_stim
            elif epoch == "choice":
                xdf_pval = xdf.p_value_choice
            if xdf_pval.iloc[0] < 0.05:
                boolean_array[idx] = True
            else:
                boolean_array[idx] = False

    return boolean_array
