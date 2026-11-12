"""
Extract VISp spike trains around stimulus onset for every BWM session with VISp units.

For each session (eid) with VISp units in the brain-wide map release:
  * load all good units (qc=1) on every probe and merge them (as in firing_rates.py)
  * keep only units whose Beryl region is VISp
  * keep only trials with a visible stimulus (contrast > 0) that pass the standard
    BWM trial mask (load_trials_and_mask defaults)
  * bin spikes in [stimOn - 0.5 s, stimOn + 0.2 s] (default 1 ms bins)
  * drop neurons that are mostly silent in that window
  * save one .npz per session to OUT_DIR/<eid>.npz

Contents of each <eid>.npz
--------------------------
eid             str
subject         str
spikes          uint8/uint16 array (n_trials, n_neurons, n_bins), spike counts per bin
bin_edges       float array (n_bins + 1,), seconds relative to stimOn
contrast        float array (n_trials,), unsigned contrast (0.0625, 0.125, 0.25, 1)
stim_side       int array (n_trials,), -1 = stimulus on the left, +1 = stimulus on the right
signed_contrast float array (n_trials,), contrast * stim_side
trial_idx       int array (n_trials,), index of the trial in the original trials table
stimOn_times    float array (n_trials,), session time of stimulus onset (s)
cluster_uuids   str array (n_neurons,)
pids            str array (n_neurons,), probe insertion of each neuron
mean_rate_hz    float array (n_neurons,), mean rate in the window across kept trials
active_frac     float array (n_neurons,), fraction of trials with >= 1 spike in the window
n_dropped       int, number of VISp neurons dropped by the silence criterion

Usage (on the server, environment info-decom):
    micromamba run -n info-decom python ibl_info/visp_stim_spiketrains.py \
        --out-dir ./data/generated/visp_stim_spiketrains --workers 8

    # test on a single session first:
    micromamba run -n info-decom python ibl_info/visp_stim_spiketrains.py --max-sessions 1
"""

import argparse
import concurrent.futures
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from one.api import ONE
from iblatlas.atlas import BrainRegions
from brainwidemap import bwm_query, bwm_units, load_good_units, load_trials_and_mask
from brainwidemap.bwm_loading import merge_probes

from ibl_info.selective_decomposition import filter_eids

REGION = "VISp"


def make_one(mode=None):
    return ONE() if mode is None else ONE(mode=mode)


def find_region_sessions(one, region=REGION):
    """Return a dataframe (eid, pid, probe_name, subject) for every BWM insertion of a
    session that has units in `region` (uses pre-written filter_eids on bwm_units)."""
    unit_df = bwm_units(one)
    eids = filter_eids(unit_df, region)
    # only the probes that actually carry region units
    region_pids = np.unique(unit_df.loc[unit_df["Beryl"] == region, "pid"])
    bwm_df = bwm_query()
    bwm_df = bwm_df[bwm_df["eid"].isin(eids) & bwm_df["pid"].isin(region_pids)]
    return bwm_df[["eid", "pid", "probe_name", "subject"]].reset_index(drop=True)


def load_region_spikes(one, eid, probe_rows, region=REGION):
    """Load good units on all listed probes, merge them and keep only `region` units."""
    spikes_list, clusters_list = [], []
    for _, row in probe_rows.iterrows():
        spk, clu = load_good_units(one, pid=row["pid"], qc=1, eid=eid, pname=row["probe_name"])
        clu = clu.copy()
        clu["pid"] = row["pid"]
        spikes_list.append(spk)
        clusters_list.append(clu)

    if len(spikes_list) > 1:
        spikes, clusters = merge_probes(spikes_list, clusters_list)
    else:
        spikes, clusters = spikes_list[0], clusters_list[0]

    beryl = BrainRegions().acronym2acronym(clusters["acronym"], mapping="Beryl")
    region_idx = np.asarray(clusters.index[beryl == region])

    keep = np.isin(spikes["clusters"], region_idx)
    times = spikes["times"][keep]
    clus = spikes["clusters"][keep]
    order = np.argsort(times, kind="stable")
    return times[order], clus[order], clusters.loc[region_idx]


def bin_aligned_spikes(times, clus, cluster_ids, align_times, t_pre, t_post, bin_size):
    """Spike counts of shape (n_trials, n_neurons, n_bins) aligned to `align_times`.
    `times` must be sorted."""
    n_bins = int(round((t_pre + t_post) / bin_size))
    bin_edges = -t_pre + np.arange(n_bins + 1) * bin_size
    lut = {c: i for i, c in enumerate(cluster_ids)}
    neuron_of = np.array([lut[c] for c in clus], dtype=np.int64) if len(clus) else np.zeros(0, int)

    counts = np.zeros((len(align_times), len(cluster_ids), n_bins), dtype=np.uint16)
    starts = np.searchsorted(times, align_times - t_pre, side="left")
    stops = np.searchsorted(times, align_times + t_post, side="left")
    for k, (a, b, t0) in enumerate(zip(starts, stops, align_times)):
        if b <= a:
            continue
        b_idx = np.floor((times[a:b] - (t0 - t_pre)) / bin_size).astype(np.int64)
        valid = (b_idx >= 0) & (b_idx < n_bins)
        np.add.at(counts[k], (neuron_of[a:b][valid], b_idx[valid]), 1)
    return counts, bin_edges


def visible_stim_trials(one, eid, use_bwm_mask=True):
    """Trials with contrast > 0 on either side (optionally after the BWM trial mask)."""
    trials, mask = load_trials_and_mask(one, eid)
    if not use_bwm_mask:
        mask = ~trials["stimOn_times"].isna()
    trials = trials[mask]
    c_left = trials["contrastLeft"].fillna(0).values
    c_right = trials["contrastRight"].fillna(0).values
    contrast = np.maximum(c_left, c_right)
    stim_side = np.where(trials["contrastRight"].notna().values, 1, -1)
    visible = contrast > 0
    return (
        trials.index.values[visible],
        trials["stimOn_times"].values[visible],
        contrast[visible],
        stim_side[visible],
    )


# --------------------------------------------------------------------------------------
# per-session worker
# --------------------------------------------------------------------------------------
def process_session(args):
    eid, probe_rows, opts = args
    out_file = Path(opts["out_dir"]) / f"{eid}.npz"
    if out_file.exists() and not opts["overwrite"]:
        return eid, "skipped (exists)"
    try:
        one = make_one(opts["one_mode"])
        trial_idx, stim_on, contrast, stim_side = visible_stim_trials(
            one, eid, use_bwm_mask=opts["use_bwm_mask"]
        )
        times, clus, region_clusters = load_region_spikes(one, eid, probe_rows)
        cluster_ids = np.asarray(region_clusters.index)
        if len(cluster_ids) == 0 or len(stim_on) == 0:
            return eid, "no VISp units or no visible-stimulus trials"

        counts, bin_edges = bin_aligned_spikes(
            times, clus, cluster_ids, stim_on, opts["t_pre"], opts["t_post"], opts["bin_size"]
        )

        # silence criterion
        window = opts["t_pre"] + opts["t_post"]
        per_trial = counts.sum(axis=2)  # (n_trials, n_neurons)
        mean_rate = per_trial.mean(axis=0) / window
        active_frac = (per_trial > 0).mean(axis=0)
        keep = (mean_rate >= opts["min_rate"]) & (active_frac >= opts["min_active_frac"])
        if keep.sum() == 0:
            return eid, f"all {len(keep)} VISp neurons silent"

        counts = counts[:, keep, :]
        if counts.max() < 256:
            counts = counts.astype(np.uint8)

        np.savez_compressed(
            out_file,
            eid=eid,
            subject=probe_rows["subject"].iloc[0],
            spikes=counts,
            bin_edges=bin_edges,
            contrast=contrast,
            stim_side=stim_side,
            signed_contrast=contrast * stim_side,
            trial_idx=trial_idx,
            stimOn_times=stim_on,
            cluster_uuids=region_clusters["uuids"].values[keep].astype(str),
            pids=region_clusters["pid"].values[keep].astype(str),
            mean_rate_hz=mean_rate[keep],
            active_frac=active_frac[keep],
            n_dropped=int((~keep).sum()),
        )
        return eid, f"saved {counts.shape[1]}/{len(keep)} neurons, {counts.shape[0]} trials"
    except Exception as e:  # keep going on failures, like the other batch scripts
        return eid, f"ERROR: {e!r}"


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", default="./data/generated/visp_stim_spiketrains")
    p.add_argument("--t-pre", type=float, default=0.5, help="seconds before stimOn")
    p.add_argument("--t-post", type=float, default=0.2, help="seconds after stimOn")
    p.add_argument("--bin-size", type=float, default=0.001, help="bin width in seconds")
    p.add_argument(
        "--min-rate",
        type=float,
        default=1.0,
        help="drop neurons whose mean rate in the window is below this (Hz)",
    )
    p.add_argument(
        "--min-active-frac",
        type=float,
        default=0.2,
        help="drop neurons that fire on fewer than this fraction of trials",
    )
    p.add_argument(
        "--no-bwm-mask",
        action="store_true",
        help="use all trials with a stimOn time instead of the standard BWM trial mask",
    )
    p.add_argument(
        "--one-mode",
        default=None,
        choices=["local", "remote", "auto"],
        help="ONE mode; default uses ONE() as in the other scripts (cached data is reused)",
    )
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    p.add_argument(
        "--max-sessions", type=int, default=None, help="process only the first N sessions"
    )
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    one = make_one(a.one_mode)
    sessions = find_region_sessions(one)
    sessions.to_csv(out_dir / "visp_sessions.csv", index=False)
    eids = list(dict.fromkeys(sessions["eid"]))
    print(f"{len(eids)} sessions ({sessions['subject'].nunique()} animals) with VISp units")
    if a.max_sessions is not None:
        eids = eids[: a.max_sessions]

    opts = dict(
        out_dir=str(out_dir),
        t_pre=a.t_pre,
        t_post=a.t_post,
        bin_size=a.bin_size,
        min_rate=a.min_rate,
        min_active_frac=a.min_active_frac,
        use_bwm_mask=not a.no_bwm_mask,
        one_mode=a.one_mode,
        overwrite=a.overwrite,
    )
    tasks = [(eid, sessions[sessions["eid"] == eid], opts) for eid in eids]

    if a.workers <= 1:
        results = [process_session(t) for t in tqdm(tasks)]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(tqdm(ex.map(process_session, tasks), total=len(tasks)))

    log = pd.DataFrame(results, columns=["eid", "status"])
    log.to_csv(out_dir / "run_log.csv", index=False)
    print(log["status"].str.split(" ").str[0].value_counts())


if __name__ == "__main__":
    main()
