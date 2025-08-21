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

config = check_config()


def region_redundancy_metrics(region_data, zscore=True):
    X = np.asarray(region_data, dtype=float)

    if zscore:
        X -= X.mean(axis=0, keepdims=True)
        stds = X.std(axis=0, ddof=1)
        stds[stds == 0] = 1.0
        X /= stds

    corr_matrix = np.corrcoef(X, rowvar=False)
    V = corr_matrix.shape[0]
    if V < 2:
        raise ValueError("Need at least 2 voxels in the region to compute correlation.")

    mean_corr = (corr_matrix.sum() - V) / (V * (V - 1))

    C = np.cov(X, rowvar=False)
    evals = np.linalg.eigvalsh(C)
    evals = np.clip(evals, 0, None)
    total = evals.sum()
    if total == 0:
        pr = 0.0
    else:
        pr = (total**2) / np.sum(evals**2)

    return float(mean_corr), float(pr)


def summary_stats(data_epoch, mask=None):
    correlation_array = np.zeros((len(data_epoch), 3))
    participation_array = np.zeros((len(data_epoch), 3))
    if mask is None:
        mask = np.ones((data_epoch[0].shape[0]), dtype=np.bool)

    for regions in range(len(data_epoch)):
        for i in range(3):
            corr, pr = region_redundancy_metrics(data_epoch[regions][mask, i, :])
            correlation_array[regions, i] = corr
            participation_array[regions, i] = pr

    return correlation_array, participation_array


def run_for_eid(session_id, regions, epoch="stim"):

    one = ONE()

    # probably this one doesnt work
    # use sessionloader
    sl = SessionLoader(one, eid=session_id)

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

    total_frames = data_epoch[0].shape[1]

    print(f"total frames: {total_frames}, regions: {actual_regions}")
    crc_all, _ = summary_stats(data_epoch)
    crc_congruent, _ = summary_stats(data_epoch, congruent_flags)
    crc_incongruent, _ = summary_stats(data_epoch, incongruent_flags)

    return crc_all, crc_congruent, crc_incongruent, actual_regions


def process_session(session_id, save_info):

    # also use regions = "single_regions" to use all
    try:
        region_pickle = run_for_eid(session_id, regions="single_regions", epoch="stim")
        with open(f"./data/generated/summary_{session_id}_wfi_{save_info}.pkl", "wb") as f:
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

    n_cores = os.cpu_count() // 2  # type: ignore

    results = Parallel(n_jobs=n_cores, verbose=10)(
        delayed(process_session)(session_id, save_info) for session_id in sessions  # type: ignore
    )

    print(results)


if __name__ == "__main__":
    run_wfi(save_info="3frames")
