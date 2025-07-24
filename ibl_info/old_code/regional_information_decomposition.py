from one.api import ONE
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from brainbox.ephys_plots import plot_brain_regions
from brainbox.behavior.wheel import velocity
from iblatlas.atlas import AllenAtlas
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from brainbox.behavior.training import compute_performance, plot_psychometric, plot_reaction_time
from brainbox.task.trials import find_trial_ids
from brainbox.io.one import SessionLoader
from pathlib import Path
from brainbox.task.trials import get_event_aligned_raster, get_psth
from brainbox import singlecell
import numpy as np
from iblatlas.atlas import BrainRegions
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from brainbox.singlecell import bin_spikes2D

from ibl_info.measures.broja_pid import compute_pid, coinformation
from ibl_info.utils import discretize_neural_data
from ibl_info.prepare_data_pid import prepare_ephys_data, compute_intervals
from brainwidemap.bwm_loading import merge_probes
from ibl_info.old_code.load_glm_hmm import load_state_dataframe


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
        u1, u2, red, syn = compute_pid(Y, X1, X2)
        coinfo, mi_yx1x2, mi_yx1, mi_yx2 = coinformation(Y, X1, X2)
        pid_information[idx, :] = u1, u2, red, syn
        coinformation_data[idx, :] = mi_yx1, mi_yx2, coinfo, mi_yx1x2

    # now to organize this?
    # nah, unique information would just be the mean of the first two
    # red and syn  are fine
    # yx1 and yx2 mutual info are also similar to UI
    # the other two are trivariate

    return pid_information, coinformation_data


def write_data(data, session_id, epoch, normalization, variable="task"):

    uuid = session_id + "_" + epoch + "_" + normalization + "_" + variable
    with open(
        f"D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\interim\{uuid}.pkl",
        "wb",
    ) as f:
        pkl.dump(data, f)


def run_decomposition(
    one, session_id, spikes, clusters, epoch, list_of_regions, normalization="neuron"
):

    sl = SessionLoader(one, eid=session_id)
    if epoch == "stim":
        trials, mask = load_trials_and_mask(
            one, session_id, exclude_nochoice=True, exclude_unbiased=False
        )
    else:
        trials, mask = load_trials_and_mask(
            one, session_id, exclude_nochoice=True, exclude_unbiased=True
        )

    trials = trials[mask]

    intervals, decoding_variable = compute_intervals(trials, epoch)
    data_epoch, actual_regions, n_units, cluster_ids = prepare_ephys_data(
        spikes, clusters, intervals, list_of_regions
    )

    # now what do we run pid here
    # why not

    # binned spikes is trials x neurons
    # the discretize function wants neurons x trials
    data = {}
    for region_idx in range(len(actual_regions)):
        data_discretized = discretize_neural_data(data_epoch[region_idx].T, normalization)
        # neurons x trials
        # now that we have discretized data, we compute the pid
        n_neurons = n_units[region_idx]
        region_name = actual_regions[region_idx]

        # NOTE: commented out for calculating neural composition
        # pid_info, coinfo = compute_information_decomposition(decoding_variable, data_discretized)
        pid_info, coinfo = neural_information_content(data_discretized)
        data[region_name] = np.hstack([pid_info, coinfo])

    # data is a dict for each region in the mice
    # each entry is a sources x 8 numpy array
    return data


def run_decompositon_glm_hmm(
    one, session_id, spikes, clusters, list_of_regions, normalization="neuron"
):

    sl = SessionLoader(one, eid=session_id)

    _, mask = load_trials_and_mask(one, session_id, exclude_nochoice=True, exclude_unbiased=False)

    K = 2
    trials_df_glm = load_state_dataframe(session_id, K=K)
    trials_df_glm["state"] = trials_df_glm[f"glm-hmm_{K}"].apply(lambda x: np.argmax(x))

    # subset it
    trials_df_glm = trials_df_glm[mask]

    for state in [0, 1]:
        trials_group = trials_df_glm[trials_df_glm["state"] == state]

        intervals, decoding_variable = compute_intervals(trials_group, "glm-hmm")
        data_epoch, actual_regions, n_units, cluster_ids = prepare_ephys_data(
            spikes, clusters, intervals, list_of_regions
        )

        data = {}
        for region_idx in range(len(actual_regions)):
            data_discretized = discretize_neural_data(data_epoch[region_idx].T, normalization)
            # neurons x trials
            # now that we have discretized data, we compute the pid
            n_neurons = n_units[region_idx]
            region_name = actual_regions[region_idx]

            # NOTE: commented out for calculating neural composition
            # pid_info, coinfo = compute_information_decomposition(decoding_variable, data_discretized)
            pid_info, coinfo = neural_information_content(data_discretized)
            data[region_name] = np.hstack([pid_info, coinfo])
        write_data(data, session_id, "prior", normalization, f"region_glm_hmm_{state}")


def neural_information_content(neural_data):
    # always same region
    # neural data is in neurons x trials
    # target is one neuron, source are the others; iterate over every neuron
    # lots of computations
    n_neurons = neural_data.shape[0]
    sources = generate_source_ids(n_neurons)

    regional_pid = np.zeros((n_neurons, 4))
    regional_coninfo = np.zeros((n_neurons, 4))

    for idx in range(n_neurons):
        # for each neuron
        targets = neural_data[idx, :]
        mask_rows = ~np.any(sources == idx, axis=1)  # drop rows with source
        new_sources = sources[mask_rows]

        # now we run for multiple sources
        pid_information = np.zeros((len(new_sources), 4))  # (neurons-{target})C2 x 4
        coinformation_data = np.zeros((len(new_sources), 4))  # (neurons-{target})C2 x 4

        # rest is same
        for idy in tqdm(range(len(new_sources))):
            s1 = sources[idy][0]
            s2 = sources[idy][1]
            X1 = np.asarray(neural_data[s1, :], dtype=np.int32)
            X2 = np.asarray(neural_data[s2, :], dtype=np.int32)
            Y = np.asarray(targets, dtype=np.int32)
            u1, u2, red, syn = compute_pid(Y, X1, X2)
            coinfo, mi_yx1x2, mi_yx1, mi_yx2 = coinformation(Y, X1, X2)
            pid_information[idx, :] = u1, u2, red, syn
            coinformation_data[idx, :] = mi_yx1, mi_yx2, coinfo, mi_yx1x2

        # compute means
        pid_information = np.mean(pid_information, axis=0)
        coinformation_data = np.mean(coinformation_data, axis=0)

        # assign to neuron
        regional_pid[idx, :] = pid_information
        regional_coninfo[idx, :] = coinformation_data

    return regional_pid, regional_coninfo


def cortical_hierarchy(one, session_id, list_of_regions, normalization="neuron"):

    pids, probes = one.eid2pid(session_id)
    if isinstance(probes, list) and len(probes) > 1:
        to_merge = [
            load_good_units(one, pid=None, eid=session_id, qc=1, pname=probe_name)
            for probe_name in probes
        ]
        spikes, clusters = merge_probes(
            [spikes for spikes, _ in to_merge], [clusters for _, clusters in to_merge]
        )
    else:
        spikes, clusters = load_good_units(one, pid=None, eid=session_id, qc=1, pname=probes)

    # for a particular epoch
    run_decompositon_glm_hmm(one, session_id, spikes, clusters, list_of_regions, normalization)

    # NOTE : the following is the correct way to run for partiuclar time intervals

    # if we want to run regional coding for prior, this is the way to go.
    # unique_epochs = ['stim','choice','feedback']

    # for epoch in tqdm(unique_epochs,leave='True',desc=f'Running for epochs'):
    #     epoch_data = run_decomposition(one, session_id, spikes, clusters, epoch, list_of_regions, normalization)
    #     write_data(epoch_data, session_id, epoch, normalization, 'neural')

    # this will generate an pid array for every group of neurons for all the eids we want


def calculate_cortical_hierachy(one, list_of_regions, list_of_eids):

    for idx in tqdm(range(len(list_of_eids)), desc="Running decomposition"):

        session_id = list_of_eids[idx]
        # cortical_hierarchy(one, session_id, list_of_regions, 'all')
        # cortical_hierarchy(one, session_id, list_of_regions, 'neuron')
        cortical_hierarchy(one, session_id, list_of_regions, "neuron")


if __name__ == "__main__":

    location = "D:\\personal\\phD\\code\\information-decomposition\\ibl-partial-info-decomp\\data\\processed\\"

    # load eids
    # nice eids covering a span of regions for decoding, glm-hmm
    list_of_eids = np.load(
        "D:\\personal\\phD\\code\\information-decomposition\\ibl-partial-info-decomp\\data\\processed\\eids_with_detailed_insertions.npy",
        allow_pickle=True,
    )

    # cortical hierarchy eids
    # list_of_eids = np.load(f'{location}minimum_cover_regions_global.npy',allow_pickle=True).item()
    # list_of_eids = np.asarray(list(list_of_eids))

    # load regions
    list_of_regions = [
        "VISpm",
        "PRNc",
        "IP",
        "VISli",
        "VM",
        "PRNc",
        "GRN",
        "VM",
        "IP",
        "APN",
        "PPN",
        "AUDp",
        "PAG",
        "PRNc",
        "IC",
        "SPVI",
        "ProS",
        "BLA",
        "SSp-ll",
        "ENTm",
        "COAp",
        "EPd",
        "ProS",
        "BLA",
        "BMA",
        "OT",
        "PA",
        "MEA",
        "EPd",
        "BLA",
    ]

    # list_of_regions = pd.read_csv(f'{location}global_hierarchy.csv').areas.values
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international")

    # setup done
    calculate_cortical_hierachy(one, list_of_regions, list_of_eids)


"""
## TODO: for ephys

1. Filter neurons before computing mutual information based on firing rate
    use ibl function for firing rate
2. Compute MI, check for significance, or set threshold (hacky)
2. Run PID only on neurons with meaningful MI
3. Check trivariate MI to see if it is more or less; must be more?
4. Block neurons by region
    then compute PID
    it will also be easy to mix and match neurons from different regions.
"""
