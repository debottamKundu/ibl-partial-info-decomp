import os
from brainwidemap.bwm_loading import load_good_units, load_trials_and_mask, merge_probes, bwm_units
import numpy as np
from brainbox.population.decode import get_spike_counts_in_bins
from brainwidemap import load_good_units, load_trials_and_mask
from brainwidemap.single_cell_stats.single_cell_util import Time_TwoNmannWhitneyUshuf
from brainwidemap.single_cell_stats.single_cell_Working_example_stimulus import (
    get_stim_time_shuffle,
)
from one.api import ONE
import pandas as pd
from tqdm import tqdm
import concurrent.futures


def BWM_stim_test_combined(one, eid, TimeWindow=np.array([0.0, 0.1])):

    # load spike data

    pids, probes = one.eid2pid(eid)
    if isinstance(probes, list) and len(probes) > 1:
        to_merge = [load_good_units(one, pid=pid, qc=1) for pid in pids]
        spikes, clusters = merge_probes(
            [spikes for spikes, _ in to_merge], [clusters for _, clusters in to_merge]
        )
    else:
        spikes, clusters = load_good_units(one, pid=pids[0], qc=1)

    # load trial data
    trials, mask = load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2.0, nan_exclude="default")
    # select good trials
    trials = trials.loc[mask == True]

    stim_on = trials.stimOn_times.to_numpy()
    contrast_R = trials.contrastRight.to_numpy()
    contrast_L = trials.contrastLeft.to_numpy()
    choice = trials.choice.to_numpy()
    block = trials.probabilityLeft.to_numpy()

    num_neuron = len(np.unique(spikes["clusters"]))
    num_trial = len(stim_on)

    ############ compute firing rate ###################

    T_1 = TimeWindow[0]
    T_2 = TimeWindow[1]

    raw_events = np.array([stim_on + T_1, stim_on + T_2]).T
    events = raw_events

    spike_count, cluster_id = get_spike_counts_in_bins(spikes["times"], spikes["clusters"], events)
    spike_rate = spike_count / (T_2 - T_1)
    area_label = clusters["atlas_id"][cluster_id].to_numpy()

    ############ return cluster id ########################
    QC_cluster_id = clusters["cluster_id"][cluster_id].to_numpy()

    ############ compute p-value for block ###################

    ########## Pre-move, time_shuffle_test #############

    p_1 = get_stim_time_shuffle(spike_rate, contrast_L, contrast_R, block, choice, 3000)

    return p_1, area_label, QC_cluster_id


def get_relevant_eids(regions_of_interest):

    one = ONE()
    unit_df = bwm_units(one)

    units_regions_of_interest = unit_df[unit_df["Beryl"].isin(regions_of_interest)]  # type: ignore
    eids = units_regions_of_interest["eid"].unique()

    return eids


def save_significance_results(eid):

    one = ONE()
    p_1, area_label, QC_cluster_id = BWM_stim_test_combined(
        one, eid, TimeWindow=np.array([0.0, 0.1])
    )
    df = pd.DataFrame(
        {
            "p_value": p_1,
            "area_label": area_label,
            "QC_cluster_id": QC_cluster_id,
        }
    )

    df.to_csv(f"./data/generated/significance_results_{eid}.csv")

    return 1


if __name__ == "__main__":

    important_regions = np.asarray(
        [
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
    )
    list_of_eids = get_relevant_eids(important_regions)

    # parallelize this
    print(f"Total tasks: {len(list_of_eids)}")

    processed_results = []
    workers = os.cpu_count() // 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results_iterator = executor.map(save_significance_results, list_of_eids)

        processed_results = list(
            tqdm(results_iterator, total=len(list_of_eids), desc="Processing Tasks")
        )
