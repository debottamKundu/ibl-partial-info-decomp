## load pairs from different regions
## from one eid
## and then do what
## i think it should more or less be the same

import itertools
import concurrent.futures
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_sample_weight
from tqdm import tqdm
from ibl_info.decoder_pid import compute_decoder_pid
from ibl_info.decoder_utils import load_specific_regions
from ibl_info.dual_decoders import complete_decoder_pid_with_null, compute_null_distribution
from ibl_info.prepare_data_pid import get_new_cinc_intervals, get_new_cinc_intervals_choice
from ibl_info.utils import check_config, epoch_events
from one.api import ONE
from prior_localization.prepare_data import prepare_widefield
from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import load_trials_and_mask
import numpy as np
import pickle as pkl
import os
from joblib import Parallel, delayed


config = check_config()


def region_combinations(n_regions):

    combinations_regions = []
    for x in itertools.combinations(range(n_regions), 2):
        combinations_regions.append([x[0], x[1]])
    combinations_regions = np.asarray(combinations_regions)
    return combinations_regions


def check_minimum(data_epoch, actual_regions, config=None):

    if config is None:
        config = check_config()

    regions_to_keep = np.zeros(len(actual_regions))
    for idx in range(len(actual_regions)):
        # print(idx)
        if data_epoch[idx].shape[-1] > config["wifi_subset"]:
            regions_to_keep[idx] = 1

    regions_to_keep = np.asarray(regions_to_keep, dtype=np.bool)
    regions_used_acronyms = np.asarray(actual_regions)[regions_to_keep]

    new_data_epoch = []

    for idx in range(len(regions_to_keep)):

        flag = regions_to_keep[idx]
        if flag == False:
            continue

        region_data = data_epoch[idx]
        if region_data.ndim == 2:
            region_data = np.expand_dims(region_data, axis=1)

        # i transpose this into frames x neurons x trials
        region_data = region_data.transpose(1, 2, 0)
        new_region_data = np.zeros_like(region_data)
        for frame in range(new_region_data.shape[0]):
            new_region_data[frame, :] = region_data[frame, :]
        new_data_epoch.append(new_region_data)

    return new_data_epoch, regions_used_acronyms


def wifi_pairs_with_all_data(eid, epoch):

    align_event = epoch_events(epoch)  # should default to stimon
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    sl = SessionLoader(one, eid=eid)
    trials, mask = load_trials_and_mask(
        one,
        eid,
        sess_loader=sl,  # using session loader to load trials so that we get proper probability
        exclude_nochoice=True,
        exclude_unbiased=True,
    )
    trials = trials[mask]

    # if only high contrast, we subset here
    # sanity check

    align_times = trials[align_event].values

    if epoch == "stim":
        intervals, target_variable, congruent_flags, incongruent_flags = get_new_cinc_intervals(
            trials, epoch
        )
    elif epoch == "choice":
        intervals, target_variable, congruent_flags, incongruent_flags = (
            get_new_cinc_intervals_choice(trials, epoch)
        )

    # run it on everything
    all_regions = config["widefield_all"]

    data_epoch, actual_regions = prepare_widefield(
        one,
        eid,
        hemisphere=config["hemisphere"],
        regions=all_regions,
        align_times=align_times,
        frame_window=config["frames"],
        functional_channel=470,
        stage_only=False,
    )

    total_frames = data_epoch[0].shape[1]  # type: ignore
    data_epoch, used_regions = check_minimum(data_epoch, actual_regions)
    region_combos = region_combinations(len(used_regions))  # type: ignore

    region_pickle = {}
    for frame_idx in range(total_frames):
        frame_pickle = {}
        for region_pairs in tqdm(region_combos, desc="region pairs "):

            region_a_idx = region_pairs[0]
            region_b_idx = region_pairs[1]

            region_a = data_epoch[region_a_idx][frame_idx, :].T
            region_b = data_epoch[region_b_idx][frame_idx, :].T

            key = f"{used_regions[region_a_idx]}_{used_regions[region_b_idx]}"

            decoder_results, null_results = complete_decoder_pid_with_null(
                target=target_variable,
                spikes_a=region_a,
                spikes_b=region_b,
                congruent_mask=congruent_flags,
                incongruent_mask=incongruent_flags,
            )

            temp = {}
            temp["decoder_results"] = decoder_results
            temp["null_results"] = null_results
            frame_pickle[key] = temp

        region_pickle[frame_idx] = frame_pickle

    return region_pickle


def wifi_pairs_of_regions(eid, epoch):

    align_event = epoch_events(epoch)  # should default to stimon
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    # probably this one doesnt work
    # use sessionloader
    sl = SessionLoader(one, eid=eid)
    trials, mask = load_trials_and_mask(
        one,
        eid,
        sess_loader=sl,  # using session loader to load trials so that we get proper probability
        exclude_nochoice=True,
        exclude_unbiased=True,
    )
    trials = trials[mask]

    # if only high contrast, we subset here
    # sanity check
    if config["high_contrast"]:
        trials.loc[:, "contrast"] = trials["contrastRight"].fillna(0) + trials[
            "contrastLeft"
        ].fillna(0)
        high_contrast_mask = trials["contrast"] >= 0.125
        trials = trials[high_contrast_mask]

    align_times = trials[align_event].values

    if epoch == "stim":
        intervals, target_variable, congruent_flags, incongruent_flags = get_new_cinc_intervals(
            trials, epoch
        )
    elif epoch == "choice":
        intervals, target_variable, congruent_flags, incongruent_flags = (
            get_new_cinc_intervals_choice(trials, epoch)
        )

    # remember there are pairs of regions now

    # load only significant decoders
    if config["decoder_over_null_filter"]:
        temp_regions = load_specific_regions(eid)
        if temp_regions == []:
            return {}
        # this is because the function needs the data in a particular way
        all_regions = [[r] for r in temp_regions]
    else:
        all_regions = config["widefield_regions"]

    data_epoch, actual_regions = prepare_widefield(
        one,
        eid,
        hemisphere=config["hemisphere"],
        regions=all_regions,
        align_times=align_times,
        frame_window=config["frames"],
        functional_channel=470,
        stage_only=False,
    )

    total_frames = data_epoch[0].shape[1]  # type: ignore
    data_epoch, used_regions = check_minimum(data_epoch, actual_regions)
    region_combos = region_combinations(len(used_regions))  # type: ignore

    region_pickle = {}
    for frame_idx in range(total_frames):
        frame_pickle = {}
        for region_pairs in tqdm(region_combos, desc="region pairs "):

            region_a_idx = region_pairs[0]
            region_b_idx = region_pairs[1]

            region_a = data_epoch[region_a_idx][frame_idx, :].T
            region_b = data_epoch[region_b_idx][frame_idx, :].T

            key = f"{used_regions[region_a_idx]}_{used_regions[region_b_idx]}"

            information_results, results = compute_decoder_pid(
                target=target_variable,
                spikes_a=region_a,
                spikes_b=region_b,
                n_bootstraps=config["n_bootstraps_decoding"],
                n_bins=config["n_bins_decoding"],
                congruent_mask=congruent_flags,
                incongruent_mask=incongruent_flags,
                decoder_output_only=config["decoder_output_only"],
            )
            temp = {}
            temp["information_results"] = information_results
            temp["results"] = results
            frame_pickle[key] = temp

        region_pickle[frame_idx] = frame_pickle

    return region_pickle


def null_pairs_of_regions(eid, epoch):
    # no discretizer or bins

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    align_event = epoch_events(epoch)  # should default to stimon
    sl = SessionLoader(one, eid=eid)
    trials, mask = load_trials_and_mask(
        one,
        eid,
        sess_loader=sl,  # using session loader to load trials so that we get proper probability
        exclude_nochoice=True,
        exclude_unbiased=True,
    )
    trials = trials[mask]
    align_times = trials[align_event].values

    if epoch == "stim":
        intervals, target_variable, congruent_flags, incongruent_flags = get_new_cinc_intervals(
            trials, epoch
        )
    elif epoch == "choice":
        intervals, target_variable, congruent_flags, incongruent_flags = (
            get_new_cinc_intervals_choice(trials, epoch)
        )
    print("Loading data")

    # remember there are pairs of regions now
    all_regions = config["widefield_regions"]
    data_epoch, actual_regions = prepare_widefield(
        one,
        eid,
        hemisphere=config["hemisphere"],
        regions=all_regions,
        align_times=align_times,
        frame_window=config["frames"],
        functional_channel=470,
        stage_only=False,
    )

    total_frames = data_epoch[0].shape[1]  # type: ignore
    data_epoch, used_regions = check_minimum(data_epoch, actual_regions)
    region_combos = region_combinations(len(used_regions))  # type: ignore

    region_pickle = {}
    for frame_idx in range(total_frames):
        frame_pickle = {}
        for region_pairs in tqdm(region_combos, desc="region pairs "):

            region_a_idx = region_pairs[0]
            region_b_idx = region_pairs[1]

            region_a = data_epoch[region_a_idx][frame_idx, :].T
            region_b = data_epoch[region_b_idx][frame_idx, :].T

            key = f"{used_regions[region_a_idx]}_{used_regions[region_b_idx]}"

            null_results = compute_null_distribution(
                neural_data_A=region_a,
                neural_data_B=region_b,
                trial_labels=target_variable,
                subset_size_D=config["wifi_subset"],
                n_permutations=config["n_bootstraps_decoding"],
            )
            frame_pickle[key] = null_results
        region_pickle[frame_idx] = frame_pickle
    return region_pickle


def process_null_distributions(session_id):
    eid = session_id
    epoch = config["epoch"]
    suffix = ""
    if config["null_computation"] == True:
        suffix += "_null"

    try:
        region_pickle = null_pairs_of_regions(eid, epoch)

        with open(f"./data/generated/{eid}_wfi_{suffix}_{epoch}.pkl", "wb") as f:
            pkl.dump(region_pickle, f)
        return 1
    except Exception as e:
        print(e)
        return -1


def process_session(session_id):

    eid = session_id
    epoch = config["epoch"]
    n_bins = config["n_bins_decoding"]
    discretizer = config["discretize_decoding"]

    suffix = f"{epoch}"

    if discretizer == 1:
        suffix += f"_equipopulated_{n_bins}"
    elif discretizer == 2:
        suffix += f"_equispaced_{n_bins}"

    if config["null_computation"] == True:
        suffix += "_null"
    if config["decoder_output_only"] == True:
        suffix += "_decoder_output"
    else:
        suffix += "_decomposition"

    if config["decoder_over_null_filter"]:
        suffix += "_significant_regions_only"

    try:
        region_pickle = wifi_pairs_of_regions(eid, epoch)

        with open(f"./data/generated/{eid}_wfi_{suffix}.pkl", "wb") as f:
            pkl.dump(region_pickle, f)
        return 1
    except Exception as e:
        print(e)
        return -1


def run_wfi_on_entire_data(session_id):

    eid = session_id
    epoch = config["epoch"]

    suffix = f"{epoch}"
    suffix += "entire_data"

    try:
        region_pickle = wifi_pairs_with_all_data(eid, epoch)

        with open(f"./data/generated/{eid}_wfi_{suffix}.pkl", "wb") as f:
            pkl.dump(region_pickle, f)
        return 1
    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    sessions = one.search(datasets="widefieldU.images.npy")
    print(f"{len(sessions)} sessions with widefield data found")  # type: ignore
    n_cores = 6  # type: ignore

    config["epoch"] = "stim"

    results = Parallel(n_jobs=n_cores, verbose=10)(delayed(run_wfi_on_entire_data)(session) for session in sessions)  # type: ignore

    config["epoch"] = "choice"
    config["frames"] = [-2, 0]
    results = Parallel(n_jobs=n_cores, verbose=10)(delayed(run_wfi_on_entire_data)(session) for session in sessions)  # type: ignore

    # results = Parallel(n_jobs=4, verbose=10)(
    #     delayed(process_null_distributions)(session) for session in sessions  # type: ignore
    # )

    # print(f"Successes: {results.count(1)}")  # type: ignore
    # print(f"Failures: {results.count(-1)}")  # type: ignore

    # config["epoch"] = "choice"
    # config["frames"] = [-2, 0]

    # results = Parallel(n_jobs=n_cores, verbose=10)(delayed(process_session)(session) for session in sessions)  # type: ignore

    # # results = Parallel(n_jobs=4, verbose=10)(
    # #     delayed(process_null_distributions)(session) for session in sessions  # type: ignore
    # # )

    # print(f"Successes: {results.count(1)}")  # type: ignore
    # print(f"Failures: {results.count(-1)}")  # type: ignore
