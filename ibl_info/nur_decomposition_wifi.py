# load different animals, run decomposition and save


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
from ibl_info.decoder_pid import compute_information_metrics
from ibl_info.utils import check_config, epoch_events, equipopulated_binning, equispaced_binning
import numpy as np
import pickle as pkl
import os
from glob import glob
from joblib import Parallel, delayed


config = check_config()


def compute_subsampled(X1_con, X2_con, Y_con, Y_incon):

    left_fraction = np.sum(Y_incon == 1) / len(Y_incon)

    # we want to ensure similar fraction for congruent subsampling
    left_congruent = np.where(Y_con == 1)[0]
    right_congruent = np.where(Y_con == 0)[0]

    sampled_information = []

    for repeats in range(5):  # should be 5 or more, lower in order to speed up

        n_left_subsample = int(np.round(left_fraction * len(Y_incon)))
        n_right_subsample = int(len(Y_incon) - n_left_subsample)

        # now we need to do the actual subsampling
        selected_indices_left = np.random.choice(left_congruent, n_left_subsample, replace=False)
        selected_indices_right = np.random.choice(
            right_congruent, n_right_subsample, replace=False
        )

        selected_indices = np.concatenate((selected_indices_left, selected_indices_right))
        subsampled_targets = Y_con[selected_indices]
        subsampled_X1 = X1_con[selected_indices]
        subsampled_X2 = X2_con[selected_indices]

        info_ = compute_information_metrics(subsampled_targets, subsampled_X1, subsampled_X2)  # type: ignore

        sampled_information.append(info_)
    sampled_information = np.asarray(sampled_information)
    sampled_information = np.mean(sampled_information, axis=0)

    return sampled_information


def run_decoder_decomposition_only(results):

    n_bootstraps = len(results)
    n_bins = config["n_bins_decoding"]

    SUBSAMPLE_FLAG = True  # i can change this

    information_array = np.zeros((n_bootstraps, 4, 7))

    for iteration in range(n_bootstraps):

        # NOTE: use this sometime

        output_a_all = results[iteration]["probs_A"]
        output_b_all = results[iteration]["probs_B"]
        target_all = results[iteration]["y_true"]

        output_a_con = results[iteration]["probs_A_cong"]
        output_b_con = results[iteration]["probs_B_cong"]
        target_con = results[iteration]["y_cong"]

        output_a_incon = results[iteration]["probs_A_incong"]
        output_b_incon = results[iteration]["probs_B_incong"]
        target_incon = results[iteration]["y_incong"]

        discretization_type = config["discretize_decoding"]
        flexible_bounds = config["flexible_bounds"]

        if discretization_type == 1:

            X1_con = np.asarray(
                equipopulated_binning(output_a_con[:, 0], n_bins=n_bins), dtype=np.int32
            )
            X2_con = np.asarray(
                equipopulated_binning(output_b_con[:, 0], n_bins=n_bins), dtype=np.int32
            )
            X1_incon = np.asarray(
                equipopulated_binning(output_a_incon[:, 0], n_bins=n_bins), dtype=np.int32
            )
            X2_incon = np.asarray(
                equipopulated_binning(output_b_incon[:, 0], n_bins=n_bins), dtype=np.int32
            )

            X1_all = np.asarray(
                equipopulated_binning(output_a_all[:, 0], n_bins=n_bins), dtype=np.int32
            )
            X2_all = np.asarray(
                equipopulated_binning(output_b_all[:, 0], n_bins=n_bins), dtype=np.int32
            )

        elif discretization_type == 2:
            X1_con = np.asarray(
                equispaced_binning(
                    output_a_con[:, 0], n_bins=n_bins, flexible_bounds=flexible_bounds
                ),
                dtype=np.int32,
            )
            X2_con = np.asarray(
                equispaced_binning(
                    output_b_con[:, 0], n_bins=n_bins, flexible_bounds=flexible_bounds
                ),
                dtype=np.int32,
            )
            X1_incon = np.asarray(
                equispaced_binning(
                    output_a_incon[:, 0], n_bins=n_bins, flexible_bounds=flexible_bounds
                ),
                dtype=np.int32,
            )
            X2_incon = np.asarray(
                equispaced_binning(
                    output_b_incon[:, 0], n_bins=n_bins, flexible_bounds=flexible_bounds
                ),
                dtype=np.int32,
            )

            X1_all = np.asarray(
                equispaced_binning(
                    output_a_all[:, 0], n_bins=n_bins, flexible_bounds=flexible_bounds
                ),
                dtype=np.int32,
            )
            X2_all = np.asarray(
                equispaced_binning(
                    output_b_all[:, 0], n_bins=n_bins, flexible_bounds=flexible_bounds
                ),
                dtype=np.int32,
            )
        else:
            raise NotImplementedError

        Y_all = np.asarray(target_all, dtype=np.int32)
        Y_con = np.asarray(target_con, dtype=np.int32)
        Y_incon = np.asarray(target_incon, dtype=np.int32)

        information_array[iteration, 0, :] = compute_information_metrics(  # type: ignore
            Y_con, X1_con, X2_con
        )

        information_array[iteration, 1, :] = compute_information_metrics(  # type: ignore
            Y_incon, X1_incon, X2_incon
        )

        information_array[iteration, 2, :] = compute_information_metrics(  # type: ignore
            Y_all, X1_all, X2_all
        )

        if SUBSAMPLE_FLAG:
            information_array[iteration, 3, :] = compute_subsampled(X1_con, X2_con, Y_con, Y_incon)

    return information_array


def process_single_animal(file):

    with open(file, "rb") as f:
        data = pkl.load(f)
    eid = file.rsplit("/")[-1].rsplit("_wfi")[0]
    region_data = {}

    epoch = "choice"
    suffix = f"{epoch}"
    n_bins = config["n_bins_decoding"]
    discretizer = config["discretize_decoding"]

    if discretizer == 1:
        suffix += f"_equipopulated_{n_bins}"
    elif discretizer == 2:
        suffix += f"_equispaced_{n_bins}"

    suffix += "_decomposition"

    try:
        for k in data[2].keys():
            results = data[2][k]["results"]
            information_array = run_decoder_decomposition_only(results)
            region_data[k] = information_array

        with open(f"./data/generated/{eid}_wfi_{suffix}.pkl", "wb") as f:
            pkl.dump(region_data, f)
        return 1
    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    files_choice = np.sort(glob("../data/generated/wfi_decoders/choice/equi_3/*.pkl"))

    results = Parallel(n_jobs=8)(
        delayed(process_single_animal)(file=f) for f in tqdm(files_choice, desc="Processing")
    )
    print(f"Successes: {results.count(1)}")  # type: ignore
    print(f"Failures: {results.count(-1)}")  # type: ignore
