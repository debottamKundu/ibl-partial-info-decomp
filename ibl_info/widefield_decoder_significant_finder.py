from joblib import Parallel, delayed
from one.api import ONE
from pathlib import Path
import numpy as np
import pandas as pd
from prior_localization.prepare_data import prepare_widefield
from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import load_trials_and_mask
from ibl_info.prepare_data_pid import get_new_cinc_intervals
import seaborn as sns
from matplotlib import pyplot as plt
from ibl_info.utils import check_config, epoch_events
from itertools import combinations
from ibl_info.utils import discretize
from ibl_info.measures import information_measures as info
import pickle as pkl
from tqdm import tqdm
import os
from ibl_info.decoder_pid import linear_nonlinear_delta
from statsmodels.stats.multitest import multipletests
import ibl_info.measures.information_measures as info
import warnings

config = check_config()


def mi_per_neuron_permuted(spikes, decoding_variable, n_permutations=100):

    mi_observed = info.corrected_mutual_information(  # type: ignore
        source=spikes, target=decoding_variable, unbiased_measure="plugin"
    )
    # use original permutation statistics
    mi_null = np.zeros(n_permutations)
    for i in range(n_permutations):

        stim_shuffled = np.random.permutation(decoding_variable)
        mi_null[i] = info.corrected_mutual_information(  # type: ignore
            source=spikes, target=stim_shuffled, unbiased_measure="plugin"
        )

    p_value = (np.sum(mi_null >= mi_observed) + 1) / (n_permutations + 1)  # type: ignore
    return mi_observed, p_value


def significant_neurons(spikes, decoding_variable, n_permutations=100, alpha=0.05):

    mi_data = np.zeros((spikes.shape[0]))
    p_values = np.zeros((spikes.shape[0]))
    for idx in tqdm(range(len(mi_data))):
        mi_data[idx], p_values[idx] = mi_per_neuron_permuted(spikes[idx, :], decoding_variable, n_permutations=n_permutations)  # type: ignore

    # do corrections
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

    return mi_data, p_values, reject


def scores_per_region(region_data, target, flags):
    # we can iterate over epochs, maybe over data_to_keep

    LLScores = []
    NLScores = []
    Diffs = []
    # for frame_idx in tqdm(range(3)):
    frame_idx = 1  # should have probably done this for 2.
    frame_data = region_data[frame_idx]
    flagged_targets = target[flags]
    linear_scores, nonlinear_scores, difference = linear_nonlinear_delta(
        flagged_targets, frame_data[flags, :]
    )
    LLScores.append(linear_scores)
    NLScores.append(nonlinear_scores)
    Diffs.append(difference)
    return LLScores, NLScores, Diffs


def wfi_linear_nonlinear_significant(session_id, regions, epoch="stim"):

    align_event = epoch_events(epoch)  # should default to stimon
    one = ONE()

    # probably this one doesnt work
    # use sessionloader
    sl = SessionLoader(one, eid=session_id)
    trials, mask = load_trials_and_mask(
        one,
        session_id,
        sess_loader=sl,  # using session loader to load trials so that we get proper probability
        exclude_nochoice=True,
        exclude_unbiased=True,
    )
    trials = trials[mask]
    align_times = trials[align_event].values
    _, target_variable, congruent_flags, incongruent_flags = get_new_cinc_intervals(trials, "stim")

    data_epoch, actual_regions = prepare_widefield(
        one,
        session_id,
        hemisphere=config["hemisphere"],
        regions=regions,
        align_times=align_times,
        frame_window=config["frames"],
        functional_channel=470,
        stage_only=False,
    )

    total_frames = data_epoch[0].shape[1]  # type: ignore
    # we will run through all frames

    # do some quick checks, don't include regions with  less than 5 voxels
    MIN_UNITS = 5
    regions_to_keep = np.zeros(len(actual_regions))  # type: ignore
    for idx in range(len(actual_regions)):  # type: ignore
        if data_epoch[idx].shape[-1] > MIN_UNITS:  # type: ignore
            regions_to_keep[idx] = 1
    regions_to_keep = np.asarray(regions_to_keep, dtype=np.bool)
    data_to_keep = np.where(regions_to_keep == 1)
    regions_used = np.asarray(actual_regions)  # just numpying it

    # do this for all regions, will take forever anyways
    congruent_region_scores = {}
    incongruent_region_scores = {}
    all_region_scores = {}
    for region_idx in tqdm(data_to_keep[0]):  # weird but whatevers
        region_data = data_epoch[region_idx].transpose(1, 0, 2)  # type: ignore
        # do this for only frame 1
        llscores, nlscores, diffs = scores_per_region(
            region_data, target_variable, congruent_flags.values
        )
        congruent_region_scores[regions_used[region_idx][0]] = [llscores, nlscores, diffs]
        llscores, nlscores, diffs = scores_per_region(
            region_data, target_variable, incongruent_flags.values
        )
        incongruent_region_scores[regions_used[region_idx][0]] = [llscores, nlscores, diffs]
        llscores, nlscores, diffs = scores_per_region(
            region_data, target_variable, np.ones(len(target_variable), dtype=bool)
        )
        all_region_scores[regions_used[region_idx][0]] = [llscores, nlscores, diffs]

    # now we do the single cell mi thing
    region_specific_mi = {}
    for region_idx in tqdm(data_to_keep[0]):  # weird but whatevers
        region_data = data_epoch[region_idx].transpose(1, 0, 2)  # type: ignore
        frx_n = []
        for frx in range(3):
            frame_we_want = region_data[frx]
            discretized_frame = discretize(
                frame_we_want.T, n_bins=4
            )  # this is 4 because we have a lot of trials for all of them
            mi_data, p_values, reject = significant_neurons(
                discretized_frame, target_variable, n_permutations=100, alpha=0.05
            )
            frx_n.append([mi_data, p_values, reject])
        region_specific_mi[regions_used[region_idx][0]] = frx_n

    information_pickle = {}
    information_pickle["congruent"] = congruent_region_scores
    information_pickle["incongruent"] = incongruent_region_scores
    information_pickle["all"] = all_region_scores
    information_pickle["single_cell_mi"] = region_specific_mi

    return information_pickle


def process_session(session_id, save_info):

    significant_regions = [
        ["MOs"],
        ["SSp-ul"],
        ["VISam"],
        ["VISl"],
        ["VISp"],
        ["ACAd"],
        ["PL"],
        ["RSPv"],
        ["VISa"],
    ]
    # also use regions = "single_regions" to use all
    try:
        region_pickle = wfi_linear_nonlinear_significant(
            session_id, regions="single_regions", epoch="stim"
        )
        with open(
            f"./data/generated/{session_id}_wfi_decoders_singlecell_{save_info}.pkl", "wb"
        ) as f:
            pkl.dump(region_pickle, f)
        return 1
    except Exception as e:
        print(f"{e}, for eid {session_id}")
        return -1


def run_wfi(save_info=""):

    one = ONE()
    sessions = one.search(datasets="widefieldU.images.npy")
    print(f"{len(sessions)} sessions with widefield data found")  # type: ignore

    # we will parallelize this
    process_session(sessions[0], save_info)  # type: ignore
    # run this commented out thing to see if it works

    # n_cores = os.cpu_count() % 2  # type: ignore

    # results = Parallel(n_jobs=n_cores, verbose=10)(
    #     delayed(process_session)(session_id, save_info) for session_id in sessions  # type: ignore
    # )

    # save this before


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_wfi(save_info="4bins")
