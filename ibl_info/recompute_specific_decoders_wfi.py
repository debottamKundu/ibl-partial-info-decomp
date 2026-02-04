from glob import glob
from ibl_info.nur_decomposition_wifi import compute_subsampled
import pandas as pd
import numpy as np
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
from ibl_info.decoder_pid import compute_decoder_pid, compute_information_metrics
from ibl_info.decoder_utils import load_specific_regions
from ibl_info.dual_decoders import complete_decoder_pid_with_null, compute_null_distribution
from ibl_info.prepare_data_pid import get_new_cinc_intervals, get_new_cinc_intervals_choice
from ibl_info.utils import check_config, epoch_events, equipopulated_binning, equispaced_binning
from one.api import ONE
from prior_localization.prepare_data import prepare_widefield
from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import load_trials_and_mask
import pickle as pkl
import os
from joblib import Parallel, delayed


config = check_config()


def get_p_value(row):
    n_better = np.sum(np.asarray(row["null_mean"]) >= row["accuracy"])
    return (n_better + 1) / (len(row["null_mean"]) + 1)


def find_significant_sessions(df):

    df["p_val"] = df.apply(get_p_value, axis=1)

    significant_sessions = df.groupby(["animal", "key_frame", "pair"]).filter(
        lambda x: (x["p_val"] < 0.05).all()
    )

    final_list = significant_sessions.pivot_table(
        index=["animal", "key_frame", "pair"],
        columns="pos",
        values="region_label",
        aggfunc="first",  # Just grabs the region name string
    ).reset_index()

    # Clean up column names
    final_list.columns.name = None  # Remove the 'pos' label from the header
    final_list = final_list.rename(columns={"First": "region_1", "Second": "region_2"})

    # Display
    print(f"Total unique pairs: {len(final_list)}")

    return final_list


def process_animal_file_for_nulls(file_path, animal_name):
    """
    Reads a pickle file and extracts balanced accuracy and null distribution means.
    """
    with open(file_path, "rb") as f:
        data = pkl.load(f)

    records = []
    for kf, regions_dict in data.items():
        for region_pair, content in regions_dict.items():
            dec = content["decoder_results"]
            nulls = content["null_results"]

            parts = region_pair.rsplit("_", 1)
            label_a, label_b = [p.replace("['", "").replace("']", "") for p in parts]

            records.append(
                {
                    "animal": animal_name,
                    "key_frame": kf,
                    "pair": region_pair,
                    "region_label": label_a,
                    "accuracy": dec.get("balanced_acc_A"),
                    "null_mean": [n.get("balanced_acc_A", 0) for n in nulls],
                    "pos": "First",
                }
            )
            records.append(
                {
                    "animal": animal_name,
                    "key_frame": kf,
                    "pair": region_pair,
                    "region_label": label_b,
                    "accuracy": dec.get("balanced_acc_B"),
                    "null_mean": [n.get("balanced_acc_B", 0) for n in nulls],
                    "pos": "Second",
                }
            )
    return records


def compute_relevant_sessions(files, frame):
    all_records = []
    for path in files:
        name = path.rsplit("/")[-1].rsplit("_")[0]
        all_records.extend(process_animal_file_for_nulls(path, name))
    df_all = pd.DataFrame(all_records)
    significant = find_significant_sessions(df=df_all)

    return significant[significant["key_frame"] == frame]


def compute_partial_id_region(results, n_bins=3):

    information_array = np.zeros((4, 7))
    n_bins = config["n_bins_decoding"]
    discretization_type = config["discretize_decoding"]

    output_a_all = results["probs_A"]
    output_b_all = results["probs_B"]
    target_all = results["y_true"]

    output_a_con = results["probs_A_cong"]
    output_b_con = results["probs_B_cong"]
    target_con = results["y_cong"]

    output_a_incon = results["probs_A_incong"]
    output_b_incon = results["probs_B_incong"]
    target_incon = results["y_incong"]

    discretization_type = config["discretize_decoding"]

    if discretization_type == 1:
        raise NotImplementedError
    elif discretization_type == 2:
        X1_con = np.asarray(equispaced_binning(output_a_con[:, 0], n_bins=n_bins), dtype=np.int32)
        X2_con = np.asarray(equispaced_binning(output_b_con[:, 0], n_bins=n_bins), dtype=np.int32)
        X1_incon = np.asarray(
            equispaced_binning(output_a_incon[:, 0], n_bins=n_bins), dtype=np.int32
        )
        X2_incon = np.asarray(
            equispaced_binning(output_b_incon[:, 0], n_bins=n_bins), dtype=np.int32
        )

        X1_all = np.asarray(equispaced_binning(output_a_all[:, 0], n_bins=n_bins), dtype=np.int32)
        X2_all = np.asarray(equispaced_binning(output_b_all[:, 0], n_bins=n_bins), dtype=np.int32)
    else:
        raise NotImplementedError

    Y_all = np.asarray(target_all, dtype=np.int32)
    Y_con = np.asarray(target_con, dtype=np.int32)
    Y_incon = np.asarray(target_incon, dtype=np.int32)

    information_array[0, :] = compute_information_metrics(Y_con, X1_con, X2_con)

    information_array[1, :] = compute_information_metrics(Y_incon, X1_incon, X2_incon)

    information_array[2, :] = compute_information_metrics(Y_all, X1_all, X2_all)

    information_array[3, :] = compute_subsampled(X1_con, X2_con, Y_con, Y_incon)

    return information_array


def compute_eid_specific_pid(data, df, frame, eid):

    region_pids = {}
    df_subset = df[(df["key_frame"] == frame) & (df["animal"] == eid)]

    region_pairs = df_subset["pair"].values

    for pair in region_pairs:
        information_array = compute_partial_id_region(data[frame][pair]["decoding_results"])
        region_pids[pair] = information_array

    return region_pids


def _process_single_file(file_path, df, frame):

    try:
        with open(file_path, "rb") as f:
            data = pkl.load(f)
        eid = file_path.rsplit("/")[-1].rsplit("_wfi")[0]
        region_pids = compute_eid_specific_pid(data, df, frame, eid)

        return (eid, region_pids)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_epoch(epoch, frame):

    if epoch == "stim":
        files = np.sort(glob("./data/generated/wfi_decoders_entire_data/allregions/stim/*.pkl"))
        signficant_df = compute_relevant_sessions(files, frame)
    elif epoch == "choice":
        files = np.sort(glob("./data/generated/wfi_decoders_entire_data/allregions/choice/*.pkl"))
        signficant_df = compute_relevant_sessions(files, frame)
    else:
        raise NotImplementedError

    # animal_results = {}
    # for idx, file in tqdm(enumerate(files), desc=f"Processing files for {epoch}"):
    #     try:
    #         with open(file, "rb") as f:
    #             data = pkl.load(f)
    #         eid = file.rsplit("/")[-1].rsplit("_wfi")[0]
    #         region_pids = compute_eid_specific_pid(data, signficant_df, frame, eid)
    #         animal_results[eid] = region_pids
    #     except Exception as e:
    #         print(e)
    #         continue

    results = Parallel(n_jobs=8, verbose=5)(
        delayed(_process_single_file)(file, signficant_df, frame) for file in files
    )

    animal_results = {res[0]: res[1] for res in results if res is not None}

    return animal_results


if __name__ == "__main__":

    epoch = "stim"
    frame = 2
    n_bins = config["n_bins_decoding"]

    animal_results_stim = process_epoch(epoch, frame)

    with open(
        f"./data/generated/wfi_animals_entire_data_stim_significant_{frame}_{n_bins}.pkl", "wb"
    ) as f:
        pkl.dump(animal_results_stim, f)

    epoch = "choice"
    frame = 2

    animal_results_choice = process_epoch(epoch, frame)
    with open(
        f"./data/generated/wfi_animals_entire_data_choice_significant_{frame}_{n_bins}.pkl", "wb"
    ) as f:
        pkl.dump(animal_results_choice, f)
