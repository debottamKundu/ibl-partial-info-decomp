from one.api import ONE
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from brainbox.ephys_plots import plot_brain_regions
from iblatlas.atlas import AllenAtlas
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from brainwidemap.bwm_loading import merge_probes
from brainbox.behavior.training import (
    compute_performance,
    plot_psychometric,
    plot_reaction_time,
    get_signed_contrast,
    compute_reaction_time,
)
from brainbox.task.trials import find_trial_ids
from brainbox.io.one import SessionLoader
from pathlib import Path
from brainbox.task.trials import get_event_aligned_raster, get_psth
from brainbox.singlecell import bin_spikes2D
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
from sklearn.ensemble import RandomForestClassifier
import os
import concurrent.futures
import functools
from ibl_info.selective_decomposition import run_analysis_single_session, filter_eids
from ibl_info.utils import check_config
import pickle as pkl
from joblib import Parallel, delayed
from tqdm import tqdm

config = check_config()


def get_accuracy_matrix(summary_df):
    """
    Converts the summary DataFrame into a 2D numpy array of accuracies.

    Returns:
    --------
    accuracy_array : np.ndarray
        Shape (N_Contrasts, 2).
        Column 0 = Congruent
        Column 1 = Incongruent
    contrasts : np.ndarray
        Shape (N_Contrasts,). The sorted contrast values corresponding to the rows.
    """
    # 1. Pivot the DataFrame
    # Index becomes the rows (Contrasts)
    # Columns become the categories (Congruent/Incongruent)
    pivot_df = summary_df.pivot(index="signed_contrast", columns="congruency", values="accuracy")

    # 2. Enforce specific column order to ensure consistency
    #    We explicitly select columns so we know Col 0 is Congruent and Col 1 is Incongruent.
    desired_order = ["Congruent", "Incongruent"]
    pivot_df = pivot_df[desired_order]

    # 3. Sort the index (contrasts) just to be safe
    pivot_df = pivot_df.sort_index()

    # 4. Convert to numpy
    accuracy_array = pivot_df.to_numpy()
    contrasts = pivot_df.index.to_numpy()

    return accuracy_array, contrasts


def get_congruency_performance(df):
    """
    Calculates the fraction of correct trials grouped by signed contrast
    and congruency (Congruent vs Incongruent).

    - Removes trials with 0.5 probability (neutral).
    - Includes 0 contrast trials by identifying their nominal side
      (based on which column is not NaN).
    """

    df_clean = df[df["probabilityLeft"] != 0.5].copy()

    has_left_stim = df_clean["contrastLeft"].notna()
    has_right_stim = df_clean["contrastRight"].notna()

    df_clean["signed_contrast"] = df_clean["contrastRight"].fillna(0) - df_clean[
        "contrastLeft"
    ].fillna(0)

    is_congruent = (has_left_stim & (df_clean["probabilityLeft"] > 0.5)) | (
        has_right_stim & (df_clean["probabilityLeft"] < 0.5)
    )

    df_clean["congruency"] = np.where(is_congruent, "Congruent", "Incongruent")

    df_clean["is_correct"] = (df_clean["feedbackType"] == 1.0).astype(int)

    summary = (
        df_clean.groupby(["signed_contrast", "congruency"])["is_correct"]
        .agg(
            accuracy="mean",
            n_trials="count",
            std_err=lambda x: x.std() / np.sqrt(len(x)),  # Standard Error
        )
        .reset_index()
    )

    return summary


def process_session(eid, one_instance):
    """
    Processes a single EID.
    Returns a tuple: (eid, result_dictionary)
    """
    try:
        # Reusing your existing logic
        trials, mask = load_trials_and_mask(
            one_instance, eid, exclude_nochoice=True, exclude_unbiased=True
        )

        # If mask or trials are empty, handle gracefully (optional safety)
        if len(trials) == 0 or mask.sum() == 0:
            return eid, None

        performance, contrasts, n_contrasts = compute_performance(trials[mask])

        # type: ignore is assumed from context
        reaction_time, _, _ = compute_reaction_time(  # type: ignore
            trials[mask], stim_off_type="firstMovement_times"
        )

        summary = get_congruency_performance(trials[mask])
        accuracy_array, contrasts = get_accuracy_matrix(summary)

        temp = {
            "performance": performance,
            "contrasts": contrasts,
            "n_contrasts": n_contrasts,
            "reaction_time": reaction_time,
            "accuracy_array": accuracy_array,
        }

        return eid, temp

    except Exception as e:
        print(f"Failed on {eid}: {e}")
        return eid, None


if __name__ == "__main__":
    important_regions = config["stim_prior_regions"]
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
    )

    unit_df = bwm_units(one)
    all_eids = []
    for region in important_regions:
        selective_eids = filter_eids(unit_df, region, significant_filter=config["decoder_filter"])
        all_eids.append(selective_eids)

    all_eids = np.concatenate(all_eids)

    all_eids = np.unique(all_eids)

    print(f"Total tasks: {len(all_eids)}")
    print(all_eids)

    jobs = (delayed(process_session)(eid, one) for eid in all_eids)
    results = Parallel(n_jobs=-1, return_as="generator")(jobs)

    info_dict = {}
    for eid, data in tqdm(results, total=len(all_eids)):  # type: ignore
        if data is not None:
            info_dict[eid] = data

    # 4. Save
    with open("./data/generated/contrastwiseperformance.pkl", "wb") as f:
        pkl.dump(info_dict, f)

    print("Done!")
