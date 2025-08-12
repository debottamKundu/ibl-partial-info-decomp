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

config = check_config()


def cleanup_and_discretize(data_epoch, actual_regions):

    # don't include regions with  less than 5 voxels
    regions_to_keep = np.zeros(len(actual_regions))
    for idx in range(len(actual_regions)):
        if data_epoch[idx].shape[-1] > config["min_units"]:
            regions_to_keep[idx] = 1

    regions_to_keep = np.asarray(regions_to_keep, dtype=np.bool)
    regions_used_acronyms = np.asarray(actual_regions)[regions_to_keep]

    discretized_data_epoch = []

    for idx in range(len(regions_to_keep)):

        flag = regions_to_keep[idx]
        if flag == False:
            continue

        # i transpose this into frames x neurons x trials
        region_data = data_epoch[idx].transpose(1, 2, 0)

        discretized_region_data = np.zeros_like(region_data)
        for frame in range(discretized_region_data.shape[0]):
            discretized_region_data[frame, :] = discretize(region_data[frame, :], n_bins=3)

        discretized_data_epoch.append(discretized_region_data)

    return discretized_data_epoch, regions_used_acronyms


def region_combinations(n_regions):

    combinations_regions = []
    for x in combinations(range(n_regions), 2):
        combinations_regions.append([x[0], x[1]])
    combinations_regions = np.asarray(combinations_regions)
    return combinations_regions


def information_for_region_pair(region_a, region_b, target):

    neurons_a = region_a.shape[0]
    neurons_b = region_b.shape[0]

    mi_region_a = np.zeros((neurons_a))
    mi_region_b = np.zeros((neurons_b))

    for n1 in tqdm(range(neurons_a)):
        mi_region_a[n1] = info.corrected_mutual_information(  # type: ignore
            target=target, source=region_a[n1, :]
        )

    for n2 in tqdm(range(neurons_b)):
        mi_region_b[n2] = info.corrected_mutual_information(  # type: ignore
            target=target, source=region_b[n2, :]
        )

    tvmi_array = np.zeros((neurons_a * neurons_b))
    pid_array = np.zeros((neurons_a * neurons_b, 4))

    combination_idx = 0

    for n1 in tqdm(range(neurons_a)):
        for n2 in range(neurons_b):

            tvmi = info.corrected_tvmi(
                source_a=region_a[n1, :],
                source_b=region_b[n2, :],
                target=target,
            )
            pid = info.corrected_pid(
                sourcea=region_a[n1, :],
                sourceb=region_b[n2, :],
                target=target,
            )

            tvmi_array[combination_idx] = tvmi  # type: ignore
            pid_array[combination_idx, :] = pid  # type: ignore

            combination_idx = combination_idx + 1

    return mi_region_a, mi_region_b, tvmi_array, pid_array


def information_computation(region_a, region_b, target):
    # region a is (neuronsxtrials)

    mi_region_a, mi_region_b, tvmi_array, pid_array = information_for_region_pair(
        region_a, region_b, target
    )

    return {
        "mi_region_a": mi_region_a,
        "mi_region_b": mi_region_b,
        "tvmi": tvmi_array,
        "pid": pid_array,
    }


def subsampled_results(region_a, region_b, target_variable, congruent_flags, incongruent_flags):

    # basic logic is lifted from the selective decomposition
    congruent_targets = target_variable[congruent_flags]
    incongruent_targets = target_variable[incongruent_flags]

    left_fraction = np.sum(incongruent_targets == 1) / len(incongruent_targets)

    # we want to ensure similar fraction for congruent subsampling
    left_congruent = np.where(congruent_targets == 1)[0]
    right_congruent = np.where(congruent_targets == 0)[0]

    sampled_mi_regiona = []
    sampled_mi_regionb = []
    sampled_pid = []
    sampled_joint = []
    for repeats in range(2):  # should be 5 or more, lower in order to speed up

        n_left_subsample = int(np.round(left_fraction * len(incongruent_targets)))
        n_right_subsample = int(len(incongruent_targets) - n_left_subsample)

        # now we need to do the actual subsampling
        selected_indices_left = np.random.choice(left_congruent, n_left_subsample, replace=False)
        selected_indices_right = np.random.choice(
            right_congruent, n_right_subsample, replace=False
        )

        selected_indices = np.concatenate((selected_indices_left, selected_indices_right))
        subsampled_targets = congruent_targets[selected_indices]
        subsampled_spikes_region_a = region_a[:, selected_indices]
        subsampled_spikes_region_b = region_b[:, selected_indices]

        info_ = information_computation(
            subsampled_spikes_region_a, subsampled_spikes_region_b, subsampled_targets
        )
        sampled_mi_regiona.append(info_["mi_region_a"])
        sampled_mi_regionb.append(info_["mi_region_b"])
        sampled_pid.append(info_["pid"])
        sampled_joint.append(info_["tvmi"])

    # average
    sampled_mi_regiona = np.asarray(sampled_mi_regiona)
    sampled_mi_regionb = np.asarray(sampled_mi_regionb)
    sampled_pid = np.asarray(sampled_pid)
    sampled_joint = np.asarray(sampled_joint)

    sampled_mi_regiona = np.mean(sampled_mi_regiona, axis=0)
    sampled_mi_regionb = np.mean(sampled_mi_regionb, axis=0)
    sampled_pid = np.mean(sampled_pid, axis=0)
    sampled_joint = np.mean(sampled_joint, axis=0)

    return {
        "mi_region_a": sampled_mi_regiona,
        "mi_region_b": sampled_mi_regionb,
        "tvmi": sampled_joint,
        "pid": sampled_pid,
    }


def wfi_by_eid(session_id, epoch="stim"):

    align_event = epoch_events(epoch)  # should default to stimon
    one = ONE()

    # probably this one doesn't work
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
        regions="single_regions",
        align_times=align_times,
        frame_window=config["frames"],
        functional_channel=470,
        stage_only=False,
    )

    total_frames = data_epoch[0].shape[1]

    # for idx in range(len(data_epoch)):
    #     print(f"{actual_regions[idx]}: , {data_epoch[idx].shape[-1]}")

    # discretize, throw away
    discretized_data, used_regions = cleanup_and_discretize(data_epoch, actual_regions)

    # now we save for pairs of regions
    # for each pair, compute congruent and incongruent conditions

    region_combos = region_combinations(len(discretized_data))

    region_pickle = {}
    for frame in range(total_frames):
        frame_pickle = {}
        # TODO: this is the best place to add parallelization
        for region_pairs in tqdm(region_combos):
            region_idx = region_pairs[0]
            region_idy = region_pairs[1]

            # now what : remove all these random things
            region_a = discretized_data[region_idx][frame, :]
            region_b = discretized_data[region_idy][frame, :]

            key = f"{str(used_regions[region_idx][0]), str(used_regions[region_idy][0])}"

            cnc_data = {}

            cnc_data["congruent"] = subsampled_results(
                region_a, region_b, target_variable, congruent_flags, incongruent_flags
            )

            cnc_data["incongruent"] = information_computation(
                region_a[:, incongruent_flags],
                region_b[:, incongruent_flags],
                target_variable[incongruent_flags],
            )

            frame_pickle[key] = cnc_data

        # so so many nested dicts
        # is there a better way to do this
        # methink no

        region_pickle[frame] = frame_pickle

    return region_pickle


def run_wfi(info=""):

    one = ONE()
    sessions = one.search(datasets="widefieldU.images.npy")
    print(f"{len(sessions)} sessions with widefield data found")  # type: ignore

    for session_id in sessions:  # type: ignore

        # run one

        try:
            region_pickle = wfi_by_eid(session_id, "stim")

            with open(f"./data/generated/{session_id}_wfi_{info}.pkl", "wb") as f:
                pkl.dump(region_pickle, f)
        except Exception as e:
            print(f"{e}, for eid {session_id}")
        break

    return


if __name__ == "__main__":
    run_wfi(info="3bins")
