from glob import glob
import itertools
import os
import pickle as pkl
from pathlib import Path
import concurrent.futures
import functools
import numpy as np
import pandas as pd
import seaborn as sns
from brainbox.behavior.training import (
    compute_performance,
    plot_psychometric,
    plot_reaction_time,
)
from brainbox.ephys_plots import plot_brain_regions
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.singlecell import bin_spikes2D
from brainbox.task.trials import find_trial_ids, get_event_aligned_raster, get_psth
from sklearn.utils import compute_sample_weight
from brainwidemap import bwm_query, bwm_units, load_good_units, load_trials_and_mask
from brainwidemap.bwm_loading import merge_probes
from iblatlas.atlas import AllenAtlas, BrainRegions
from matplotlib import pyplot as plt
from one.api import ONE
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from ibl_info.decoder_utils import return_congruent_incongruent_flags
from ibl_info.selective_decomposition import filter_eids
from sklearn.svm import SVC
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
import ibl_info.measures.information_measures as info
from ibl_info.prepare_data_pid import (
    get_new_cinc_intervals,
    prepare_ephys_data,
    get_new_cinc_intervals_choice,
)
from ibl_info.utils import (
    check_config,
    equipopulated_binning,
    equispaced_binning,
)
from ibl_info.decoder_pid import compute_information_metrics
from joblib import Parallel, delayed
from scipy.stats import wilcoxon


def recompute_with_discretization_on_all(data, epoch, n_bins):

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    recomputed_data = {}
    for animal_id in tqdm(data.keys(), desc="Animals"):
        animal = data[animal_id]
        results = animal["decoding_results"]
        n_bootstraps = len(results)

        congruent_flags, incongruent_flags = return_congruent_incongruent_flags(
            one,
            animal_id,
            epoch,
        )

        information_array = np.zeros((n_bootstraps, 2, 7))

        temp = {}
        for iteration in tqdm(range(n_bootstraps), desc="Bootstraps", leave=False):
            output_a_all = results[iteration]["probs_A"]
            output_b_all = results[iteration]["probs_B"]
            target_all = results[iteration]["y_true"]

            target_con = results[iteration]["y_cong"]
            target_incon = results[iteration]["y_incong"]
            discretization_type = config["discretize_decoding"]

            if discretization_type == 3:
                equipop_output_a = equipopulated_binning(output_a_all[:, 0], n_bins=n_bins)
                equipop_output_b = equipopulated_binning(output_b_all[:, 0], n_bins=n_bins)
            elif discretization_type == 4:
                equipop_output_a = equispaced_binning(output_a_all[:, 0], n_bins=n_bins)
                equipop_output_b = equispaced_binning(output_b_all[:, 0], n_bins=n_bins)
            else:
                raise NotImplementedError

            X1_con = np.asarray(equipop_output_a[congruent_flags], dtype=np.int32)
            X2_con = np.asarray(equipop_output_b[congruent_flags], dtype=np.int32)

            X1_incon = np.asarray(equipop_output_a[incongruent_flags], dtype=np.int32)
            X2_incon = np.asarray(equipop_output_b[incongruent_flags], dtype=np.int32)

            Y_con = np.asarray(target_con, dtype=np.int32)
            Y_incon = np.asarray(target_incon, dtype=np.int32)

            information_array[iteration, 0, :] = compute_information_metrics(  # type: ignore
                Y_con, X1_con, X2_con
            )

            information_array[iteration, 1, :] = compute_information_metrics(  # type: ignore
                Y_incon, X1_incon, X2_incon
            )
            temp["information"] = information_array
        recomputed_data[animal_id] = temp

    return recomputed_data


def process_file(filename, n_bins):

    try:
        data = load_pickle(filename)
        if config["subset"]:
            recomputed_data = recompute_with_discretization_on_all(data, config["epoch"], n_bins)
        else:
            recomputed_data = recompute(data, n_bins)
        epoch = config["epoch"]
        region_name = filename.rsplit(f"_{epoch}")[0].rsplit("_")[-1]
        if config["discretize_decoding"] == 1:
            decoding = "equipopulated"
        elif config["discretize_decoding"] == 2:
            decoding = "equispaced"
        elif config["discretize_decoding"] == 3:
            decoding = "equipop_subset"
        elif config["discretize_decoding"] == 4:
            decoding = "equispaced_subset"
        else:
            raise NotImplementedError
        with open(
            f"./data/generated/recomputed_{region_name}_{epoch}_{decoding}_{n_bins}.pkl",
            "wb",
        ) as f:
            pkl.dump(recomputed_data, f)
        return (True, f"Processed: {region_name}")
    except Exception as e:
        return (False, f"FAILED {filename}: {str(e)}")


def save_recomputed_data_parallel(files, n_bins):

    total_cores = os.cpu_count()

    workers = total_cores // 2  # type: ignore
    results = Parallel(n_jobs=workers)(delayed(process_file)(f, n_bins) for f in tqdm(files))

    failures = [msg for success, msg in results if not success]  # type: ignore

    if failures:
        print(f"Process completed with {len(failures)} errors:")
        for fail_msg in failures:
            print(fail_msg)
    else:
        print("Success: All files processed without errors.")


config = check_config()


def load_pickle(file):

    with open(file, "rb") as f:
        return pkl.load(f)


def recompute(data, n_bins=3):
    recomputed_data = {}
    for animal_id in tqdm(data.keys(), desc="Animals"):
        animal = data[animal_id]
        results = animal["decoding_results"]
        n_bootstraps = len(results)

        information_array = np.zeros((n_bootstraps, 3, 7))
        for iteration in tqdm(range(n_bootstraps), desc="Bootstraps", leave=False):
            temp = {}

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

            # if discretization_type == 1:

            #     X1_con = np.asarray(
            #         equipopulated_binning(output_a_con[:, 0], n_bins=n_bins), dtype=np.int32
            #     )
            #     X2_con = np.asarray(
            #         equipopulated_binning(output_b_con[:, 0], n_bins=n_bins), dtype=np.int32
            #     )
            #     X1_incon = np.asarray(
            #         equipopulated_binning(output_a_incon[:, 0], n_bins=n_bins), dtype=np.int32
            #     )
            #     X2_incon = np.asarray(
            #         equipopulated_binning(output_b_incon[:, 0], n_bins=n_bins), dtype=np.int32
            #     )
            if discretization_type == 2:
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
            temp["information"] = information_array
        recomputed_data[animal_id] = temp

    return recomputed_data


# this changes, we recompute with flexile equispaced binnings
# save results
if __name__ == "__main__":

    # allsessions, goodsessions
    # stim and choice

    location = "/usr/people/kundu/code/ibl-partial-info-decomp/data/generated/pairwise_decoders/stim/allsessions/equidistant_5bins/"
    files = glob(f"{location}/*.pkl")

    config["epoch"] = "stim"
    config["discretize_decoding"] = 2  # this is equispaced

    save_recomputed_data_parallel(files, 3)
    save_recomputed_data_parallel(files, 4)

    # chance locations and rerun

    location = "/usr/people/kundu/code/ibl-partial-info-decomp/data/generated/pairwise_decoders/choice/allsessions/equipopulated_5/"
    files = glob(f"{location}/*.pkl")
    config["epoch"] = "choice"

    save_recomputed_data_parallel(files, 3)
    save_recomputed_data_parallel(files, 4)

    location = "/usr/people/kundu/code/ibl-partial-info-decomp/data/generated/pairwise_decoders/stim/goodsessions/equidistant_5bins/"
    files = glob(f"{location}/*.pkl")
    config["epoch"] = "stim"

    save_recomputed_data_parallel(files, 3)
    save_recomputed_data_parallel(files, 4)

    location = "/usr/people/kundu/code/ibl-partial-info-decomp/data/generated/pairwise_decoders/choice/goodsessions/equipopulated_5/"
    files = glob(f"{location}/*.pkl")
    config["epoch"] = "choice"

    save_recomputed_data_parallel(files, 3)
    save_recomputed_data_parallel(files, 4)
