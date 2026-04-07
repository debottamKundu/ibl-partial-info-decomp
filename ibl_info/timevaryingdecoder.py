import numpy as np
from one.api import ONE
from brainwidemap import load_good_units, load_trials_and_mask
from brainwidemap.bwm_loading import merge_probes
from brainbox.singlecell import bin_spikes2D
from iblatlas.regions import BrainRegions
from scipy.ndimage import convolve1d
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import pickle as pkl
import concurrent.futures
import numpy as np
from tqdm import tqdm
from one.api import ONE

from brainwidemap import bwm_units, load_good_units, load_trials_and_mask
from brainwidemap.bwm_loading import merge_probes
from brainbox.singlecell import bin_spikes2D
from iblatlas.regions import BrainRegions
from scipy.ndimage import convolve1d

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ibl_info.selective_decomposition import filter_eids
from ibl_info.utils import check_config

config = check_config()


def run_time_varying_decoder(
    session_id,
    region,
    align_event="stimOn_times",
    t_pre=0.5,
    t_post=0.0,
    bin_size=0.05,
    stride=0.01,
):
    """
    Runs a time-resolved logistic regression decoder for a specific region.
    """
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    pids, probes = one.eid2pid(session_id)
    if isinstance(probes, list) and len(probes) > 1:
        to_merge = [load_good_units(one, pid=pid, qc=1) for pid in pids]
        spikes, clusters = merge_probes(
            [spikes for spikes, _ in to_merge], [clusters for _, clusters in to_merge]
        )
    else:
        spikes, clusters = load_good_units(one, pid=pids[0], qc=1)

    trials, mask = load_trials_and_mask(
        one, session_id, exclude_nochoice=True, exclude_unbiased=True
    )
    trials = {k: v[mask] for k, v in trials.items()}

    y = trials["feedbackType"]

    br = BrainRegions()
    acronyms = br.id2acronym(clusters["atlas_id"], mapping="Beryl")
    in_region = np.isin(acronyms, [region])

    if np.sum(in_region) < 10:  # Minimum unit threshold safety check
        print(f"Not enough units in {region}.")
        return None, None

    target_ids = clusters["cluster_id"][in_region]
    all_spike_ids = clusters["cluster_id"][spikes["clusters"]]
    spike_mask = np.isin(all_spike_ids, target_ids)

    region_spike_times = spikes["times"][spike_mask]
    region_spike_ids = all_spike_ids[spike_mask]

    align_times = trials[align_event].values - 0.1  #

    binned, times = bin_spikes2D(
        region_spike_times,
        region_spike_ids,
        target_ids,
        align_times,
        t_pre,
        t_post,  # type: ignore
        bin_size,
    )

    # w_points = int(bin_size / stride)
    # kernel = np.ones(w_points) / w_points
    # smoothed_binned = convolve1d(binned, kernel, axis=-1, mode="nearest")

    # --- 4. Decoding Loop across Time (from outcome_decoder.py) ---
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000),
            ),
        ]
    )

    cv_rule = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    n_timebins = binned.shape[2]
    time_scores = np.zeros(n_timebins)

    print(f"Running decoder across {n_timebins} time bins for {region}...")
    for t in range(n_timebins):

        X_t = binned[:, :, t]

        # Cross-validate
        scores = cross_val_score(
            estimator=pipeline, X=X_t, y=y, cv=cv_rule, scoring="balanced_accuracy", n_jobs=-1
        )
        time_scores[t] = np.mean(scores)

    return times, time_scores


def _worker_wrapper(args):
    """Unpacks arguments for the ProcessPoolExecutor."""
    eid, region, align_event, t_pre, t_post, bin_size, stride = args
    try:
        times, scores = run_time_varying_decoder(
            eid, region, align_event, t_pre, t_post, bin_size, stride
        )
        return region, eid, times, scores
    except Exception as e:
        print(f"Error in {eid} for {region}: {e}")
        return region, eid, None, None


def run_parallel_time_varying(list_of_regions, align_event="stimOn_times", t_pre=0.5, t_post=0.0):
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )

    unit_df = bwm_units(one)
    all_tasks_to_run = []

    print("Gathering sessions for regions...")
    for region in list_of_regions:
        selective_eids = filter_eids(unit_df, region, significant_filter=config["decoder_filter"])
        for eid in selective_eids:
            # Append arguments needed for the worker
            all_tasks_to_run.append((eid, region, align_event, t_pre, t_post, 0.05, 0.01))

    print(f"Total tasks (Region-Session pairs): {len(all_tasks_to_run)}")

    # Execute in parallel
    workers = max(1, os.cpu_count() // 4)
    processed_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results_iterator = executor.map(_worker_wrapper, all_tasks_to_run)

        processed_results = list(
            tqdm(
                results_iterator, total=len(all_tasks_to_run), desc="Running Time-Varying Decoders"
            )
        )

    # Aggregate and Save Results
    print("Aggregating and saving results...")
    region_data = {region: {} for region in list_of_regions}

    for region, eid, times, scores in processed_results:
        if scores is not None and times is not None:
            region_data[region][eid] = {"times": times, "scores": scores}

    # Save one pickle per region
    os.makedirs("./data/generated", exist_ok=True)
    for region, data_dict in region_data.items():
        if data_dict:  # Only save if we have successful runs for this region
            save_path = f"./data/generated/time_varying_{region}_{align_event}.pkl"
            with open(save_path, "wb") as f:
                pkl.dump(data_dict, f)
            print(f"Saved {len(data_dict)} sessions to {save_path}")

    print("Finished parallel execution!")


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

    run_parallel_time_varying(important_regions, align_event="stimOn_times", t_pre=0.5, t_post=0.0)
