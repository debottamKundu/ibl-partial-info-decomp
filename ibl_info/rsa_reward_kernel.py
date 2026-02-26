import concurrent.futures
import pickle as pkl
import time
from joblib import Parallel, delayed
from one.api import ONE
import pandas as pd
from tqdm import tqdm
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from brainbox.singlecell import bin_spikes2D
from iblatlas.regions import BrainRegions
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ibl_info.pseudosession import get_requisite_eids
from ibl_info.utils import check_config, compute_animal_stats, get_trial_masks_detailed
from scipy.ndimage import convolve1d
import traceback
from scipy.stats import zscore
from ibl_info.manifold import get_trial_masks, get_action_kernel_congruence

config = check_config()
MY_REGIONS = config["stim_prior_regions"]
MIN_NEURONS = config["min_units"]


def get_preceding_reward_history(trials, masks, n=1):
    """
    Computes the preceding reward history for different trial conditions.
    """

    is_correct = (trials["feedbackType"] == 1).astype(float)
    rolling_correct_sum = is_correct.shift(1).rolling(window=n, min_periods=1).sum()
    rolling_correct_mean = is_correct.shift(1).rolling(window=n, min_periods=1).mean()

    results = {}
    for condition_name, mask in masks.items():

        avg_sum_for_cond = rolling_correct_sum[mask].mean()
        avg_mean_for_cond = rolling_correct_mean[mask].mean()

        if n == 1:
            results[condition_name] = {"avg_number_of_correct_preceding_n": avg_sum_for_cond}
        else:
            results[condition_name] = {
                "avg_number_of_correct_preceding_n": avg_sum_for_cond,
                "avg_proportion_correct_preceding_n": avg_mean_for_cond,
            }

    return results


def process_session(eid, one, only_correct=True, simpler_mask=False):
    """Function containing the logic for a single session."""
    try:
        trials, trial_mask = load_trials_and_mask(
            one, eid, exclude_unbiased=True, exclude_nochoice=True
        )

        trials = trials[trial_mask]

        if simpler_mask:
            cond_masks, cond_names = get_trial_masks(trials)
        else:
            cond_masks, cond_names = get_action_kernel_congruence(
                eid, trial_mask, only_corr=only_correct
            )

        condition_averages = get_preceding_reward_history(trials, cond_masks, n=5)

        return (eid, condition_averages)

    except Exception as e:
        print(f"Error processing {eid}: {e}")
        return (eid, None)


if __name__ == "__main__":

    important_regions = [
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

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
    )
    global_eid_list = get_requisite_eids(one, important_regions)

    # only_correct = True
    # results_list = Parallel(n_jobs=-1)(
    #     delayed(process_session)(eid, one, only_correct) for eid in global_eid_list
    # )

    # big_dict = {eid: df for eid, df in results_list if df is not None}  # type: ignore

    # with open(
    #     "./data/processed/all_eids_dict_for_preceding_reward_history_correct.pkl", "wb"
    # ) as f:
    #     pkl.dump(big_dict, f)

    # only_correct = False
    # results_list = Parallel(n_jobs=-1)(
    #     delayed(process_session)(eid, one, only_correct) for eid in global_eid_list
    # )

    # big_dict = {eid: df for eid, df in results_list if df is not None}  # type: ignore

    # with open(
    #     "./data/processed/all_eids_dict_for_preceding_reward_history_all_conditions.pkl", "wb"
    # ) as f:
    #     pkl.dump(big_dict, f)

    # i also want to use the old function

    # use simpler mask
    only_correct = True
    use_simpler = True

    results_list = Parallel(n_jobs=-1)(
        delayed(process_session)(eid, one, only_correct, use_simpler) for eid in global_eid_list
    )

    big_dict = {eid: df for eid, df in results_list if df is not None}  # type: ignore

    with open(
        "./data/processed/all_eids_dict_for_preceding_reward_history_older_conditions.pkl", "wb"
    ) as f:
        pkl.dump(big_dict, f)
