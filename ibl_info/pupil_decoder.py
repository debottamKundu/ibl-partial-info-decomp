import concurrent.futures
import pickle as pkl
import time
from one.api import ONE
import pandas as pd
from tqdm import tqdm
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from brainbox.singlecell import bin_spikes2D
from iblatlas.regions import BrainRegions
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ibl_info.utils import check_config, compute_animal_stats
from scipy.ndimage import convolve1d
import traceback
from scipy.stats import zscore
from brainbox.io.one import SessionLoader

config = check_config()
MY_REGIONS = config["stim_prior_regions"]


def extract_baseline_pupil(trials_df, pupil_df, window=(-0.6, -0.1)):
    """
    Extracts the average pupil diameter for a specific window relative to Stimulus Onset.

    Parameters:
    - trials_df: DataFrame with 'stimOn_times'
    - pupil_df: DataFrame with 'times' and 'pupil_diameter' (or 'smooth')
    - window: tuple (start_offset, end_offset) e.g., (-0.6, -0.1)

    Returns:
    - trials_df with a new column 'pupil_baseline' (Z-scored per session is recommended later)
    """

    if not pupil_df["times"].is_monotonic_increasing:
        pupil_df = pupil_df.sort_values("times")

    p_times = pupil_df["times"].values
    p_vals = pupil_df["pupilDiameter_smooth"].values

    stim_times = trials_df["stimOn_times"].values
    starts = stim_times + window[0]
    ends = stim_times + window[1]

    idx_start = np.searchsorted(p_times, starts)
    idx_end = np.searchsorted(p_times, ends)

    temp_means = []
    for i, j in zip(idx_start, idx_end):
        if i < j:
            chunk = p_vals[i:j]
            temp_means.append(np.nanmean(chunk))
        else:
            temp_means.append(np.nan)
    # 5. Add to DataFrame
    trials_df = trials_df.copy()
    trials_df["pupil_baseline_raw"] = temp_means

    return trials_df


def process_pupil_data(trials, pupil, window):
    """
    Wrapper to handle the Z-Scoring logic correctly.
    Crucial: Z-Score must be done PER MOUSE/SESSION, not globally.
    """
    trials = extract_baseline_pupil(trials, pupil, window=window)

    vals = trials["pupil_baseline_raw"].values

    mean_val = np.nanmean(vals)
    std_val = np.nanstd(vals)

    trials["pupil_z"] = (vals - mean_val) / std_val

    return trials


def create_global_df(eid_list, window=(-0.6, -0.1)):

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    global_df = []
    for eid in tqdm(eid_list):
        try:
            sl = SessionLoader(one, eid)
            trials, mask = load_trials_and_mask(
                one, eid, exclude_unbiased=True, exclude_nochoice=True
            )
            trials = trials[mask]
            sl.load_pupil()
            trials = process_pupil_data(trials, sl.pupil, window)  # type: ignore
            keep_columns = [
                "feedbackType",
                "probabilityLeft",
                "contrastRight",
                "contrastLeft",
                "pupil_z",
            ]
            trials = trials[keep_columns]
            trials["eid"] = eid
            global_df.append(trials)
        except Exception as e:
            print(f"Error in {eid}: {e}")
            traceback.print_exc()
    return pd.concat(global_df, ignore_index=True)


if __name__ == "__main__":

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    print("Querying BWM Units...")

    units_df = bwm_units(one)
    relevant_pids = units_df[units_df["Beryl"].isin(MY_REGIONS)]["pid"].unique()

    bwm_df = bwm_query(one)
    subset_df = bwm_df[bwm_df["pid"].isin(relevant_pids)]

    task_list = [(row["pid"], row["eid"]) for _, row in subset_df.iterrows()]

    list_of_eids = subset_df["eid"].unique()

    df_all = create_global_df(list_of_eids, window=(-0.6, -0.1))

    df_all.to_csv("./data/generated/pupil_stats_largerinterval.csv", index=False)

    df_all = create_global_df(list_of_eids, window=(-0.6, -0.5))
    df_all.to_csv("./data/generated/pupil_stats_smallerinterval.csv", index=False)
