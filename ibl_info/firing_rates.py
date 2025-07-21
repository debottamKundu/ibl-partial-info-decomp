from ibl_info.prepare_data_pid import (
    get_congruent_incongruent_intervals,
    get_window,
    get_contrast_intervals,
)
from ibl_info.selective_decomposition import filter_eids
import numpy as np
from brainbox.population.decode import get_spike_counts_in_bins
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from brainbox.ephys_plots import plot_brain_regions
from iblatlas.atlas import AllenAtlas
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from brainwidemap.bwm_loading import merge_probes
from brainbox.behavior.training import compute_performance, plot_psychometric, plot_reaction_time
from brainbox.task.trials import find_trial_ids
from brainbox.io.one import SessionLoader
from pathlib import Path
from brainbox.singlecell import bin_spikes2D, firing_rate
import numpy as np
from iblatlas.atlas import BrainRegions
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
import warnings
import os
import concurrent.futures
import time


def get_spike_data_in_binsv2(spikes, clusters, intervals, region):

    # do check if cluster acronyms are in the regions provided
    brainreg = BrainRegions()
    beryl_regions = brainreg.acronym2acronym(clusters["acronym"], mapping="Beryl")

    # find all clusters in region (where region can be a list of regions)
    region_mask = np.isin(beryl_regions, region)
    actual_regions = region
    n_units = np.sum(region_mask)

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

    global_firing_rate = clusters["firing_rate"][region_mask].values

    return binned, actual_regions, n_units, clusters_uuids, global_firing_rate


def get_firing_rates_for_animal(one, session_id, epoch, region):

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

    # for now we are looking at just (stimulus interval)
    # we know the order
    intervals_by_congruency, _ = get_congruent_incongruent_intervals(trials, epoch)
    intervals_by_contrast = get_contrast_intervals(trials, epoch)
    # also compute firing rate using the entire window

    binned_spikes = []
    for interval in intervals_by_congruency:
        binned, _, _, _, firing_rate_a = get_spike_data_in_binsv2(
            spikes, clusters, interval, region
        )
        binned_spikes.append(binned)

    for interval in intervals_by_contrast:
        binned, _, _, _, firing_rate_b = get_spike_data_in_binsv2(
            spikes, clusters, interval, region
        )
        binned_spikes.append(binned)

    # firing rate a should be the same as firing rate b
    assert np.allclose(firing_rate_a, firing_rate_b)  # type: ignore
    spike_rates = np.zeros(
        (len(binned_spikes) + 1, len(binned_spikes[0]))
    )  # one to add the global rate
    time_window = get_window("stim")

    for i, binned_neuron in enumerate(binned_spikes):
        binned_rate = binned_neuron / time_window[1] - time_window[0]
        spike_rates[i, :] = np.mean(binned_rate, axis=1)

    spike_rates[-1, :] = firing_rate_b  # type: ignore

    return spike_rates


def prepare_and_run_data(args):
    eid, region, epoch = args
    one = ONE()
    try:
        spike_rates = get_firing_rates_for_animal(one, eid, epoch, region)
        return spike_rates, eid, region
    except Exception as e:
        print(e)
        return None, eid, region


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

    # region, then eid
    one = ONE()
    unit_df = bwm_units(one)
    all_tasks_to_run = []
    for region in important_regions:

        selective_eids = filter_eids(unit_df, region)
        for eid in tqdm(selective_eids):
            all_tasks_to_run.append((eid, region, "stim"))
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
    for spike_data, region, eid in processed_results:

        if region not in region_data:
            region_data[region] = {}

        if spike_data is not None:
            region_data[region][eid] = spike_data

    # this will make one huge pickle:
    # with regions and then eids for each region

    # now we can iterate through items and save
    for region, region_pickle in region_data.items():
        with open(f"./data/generated/spiking_rate_{region}_stim.pkl", "wb") as f:
            pkl.dump(region_pickle, f)

    print("Done!")
