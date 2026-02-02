import concurrent.futures
import pickle as pkl
import time
from one.api import ONE
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from brainbox.singlecell import bin_spikes2D
from iblatlas.regions import BrainRegions
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ibl_info.utils import check_config
from scipy.ndimage import convolve1d
import traceback

config = check_config()
CORRECT = False
MY_REGIONS = config["stim_prior_regions"]
MIN_NEURONS = config["min_units"]
BIN_SIZE = 0.01  # 10ms
STRIDE = 0.001  # 1ms
USE_SLIDING_WINDOW = config["use_sliding_window"]


EPOCHS = {
    "Quiescent": {
        "align": "stimOn_times",
        "offset": -0.1,  # Align to -0.1s before Stim
        "t_pre": 0.5,
        "t_post": 0.0,
    },
    "Stimulus": {"align": "stimOn_times", "offset": 0.0, "t_pre": 0.0, "t_post": 0.15},
    "Choice": {"align": "firstMovement_times", "offset": 0.0, "t_pre": 0.15, "t_post": 0.0},
}

COND_NAMES = ["L_Cong", "L_Incong", "R_Cong", "R_Incong"]
COLORS = ["#00008B", "#6495ED", "#8B0000", "#FA8072"]


def get_trial_masks(trials, correct=True):
    """
    Returns boolean masks for conditions.
    Strictly filters for CORRECT trials (feedbackType == 1).
    """
    masks = {}

    has_contrast_L = ~np.isnan(trials["contrastLeft"])
    has_contrast_R = ~np.isnan(trials["contrastRight"])

    is_L_block = trials["probabilityLeft"] == 0.8
    is_R_block = trials["probabilityLeft"] == 0.2
    is_correct = trials["feedbackType"] == 1

    if not correct:
        is_correct = ~is_correct  # essentially wrong

    masks["L_Cong"] = has_contrast_L & is_L_block & is_correct
    masks["L_Incong"] = has_contrast_L & is_R_block & is_correct
    masks["R_Cong"] = has_contrast_R & is_R_block & is_correct
    masks["R_Incong"] = has_contrast_R & is_L_block & is_correct

    return masks


def calculate_euclidean_dist(vec_a, vec_b):
    """Calculates dist between two population vectors (axis=1 is neurons)."""
    return np.linalg.norm(vec_a - vec_b, axis=1)


def process_single_session(
    pid,
    eid,
    requested_regions,
    epochs_config,
    use_slide,
    win_size,
    stride,
    bin_simple,
    correct=True,
):
    """
    Loads one session, extracts spikes for requested regions, and computes PETHs.
    """

    one_local = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    br_local = BrainRegions()
    session_results = {}

    try:

        spikes, clusters = load_good_units(one_local, pid)
        trials, mask = load_trials_and_mask(one_local, eid)
        trials = {k: v[mask] for k, v in trials.items()}

        all_spike_ids = clusters["cluster_id"][spikes["clusters"]]

        masks = get_trial_masks(trials, correct)

        if np.sum(masks["Left"]) < 5 or np.sum(masks["Right"]) < 5:
            return None

        acronyms = br_local.id2acronym(clusters["atlas_id"], mapping="Beryl")

        for region in requested_regions:

            in_region = np.isin(acronyms, [region])
            if np.sum(in_region) < MIN_NEURONS:
                continue

            target_ids = clusters["cluster_id"][in_region]

            spike_mask = np.isin(all_spike_ids, target_ids)
            region_spike_times = spikes["times"][spike_mask]
            region_spike_ids = all_spike_ids[spike_mask]

            session_results[region] = {}

            for epoch_name, params in epochs_config.items():
                epoch_stack = []
                offset = params.get("offset", 0.0)

                for cond in COND_NAMES:
                    if np.sum(masks[cond]) < 1:
                        return None

                    base_times = trials[params["align"]][masks[cond]].values
                    align_times = base_times + offset

                    if use_slide:

                        binned, _ = bin_spikes2D(
                            region_spike_times,
                            region_spike_ids,
                            target_ids,
                            align_times,
                            params["t_pre"],
                            params["t_post"],
                            stride,
                        )

                        w_points = int(win_size / stride)
                        kernel = np.ones(w_points) / w_points
                        smoothed = convolve1d(binned, kernel, axis=-1, mode="nearest")

                        psth = np.mean(smoothed, axis=0)
                    else:

                        binned, _ = bin_spikes2D(
                            region_spike_times,
                            region_spike_ids,
                            target_ids,
                            align_times,
                            params["t_pre"],
                            params["t_post"],
                            bin_simple,
                        )
                        psth = np.mean(binned, axis=0)

                    epoch_stack.append(psth)

                session_results[region][epoch_name] = np.hstack(epoch_stack)

        return session_results

    except Exception as e:
        print(e)
        return None


def plot_pcas_and_euclids(accumulated_data):

    for region in MY_REGIONS:
        # if not any(accumulated_data[region]["Stimulus"]):
        #     print(f"Skipping {region} (Insufficient Data)")
        #     continue

        print(f"\n--- Visualizing Region: {region} ---")

        epochs_ordered = ["Quiescent", "Stimulus", "Choice"]

        fig1 = plt.figure(figsize=(15, 5))
        fig1.suptitle(f"{region} | PCA State Space", fontsize=16)

        for i, epoch_name in enumerate(epochs_ordered):
            session_matrices = accumulated_data[region][epoch_name]
            if not session_matrices:
                continue

            pop_matrix = np.vstack(session_matrices)

            # Standard PCA on the 4 conditions
            pca = PCA(n_components=3)
            X_embedded = pca.fit_transform(pop_matrix.T)

            n_bins = int(X_embedded.shape[0] / 4)
            trajs = X_embedded.reshape(4, n_bins, 3)

            ax = fig1.add_subplot(1, 3, i + 1, projection="3d")

            # Plot the 4 Conditions
            for c in range(4):
                ax.plot(
                    trajs[c, :, 0],
                    trajs[c, :, 1],
                    trajs[c, :, 2],
                    c=COLORS[c],
                    label=COND_NAMES[c],
                    lw=2,
                    alpha=0.9,
                )
                ax.scatter(trajs[c, 0, 0], trajs[c, 0, 1], trajs[c, 0, 2], color=COLORS[c], s=10)

            ax.set_title(epoch_name)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            if i == 2:
                ax.legend(loc="upper right", fontsize="xx-small")

        plt.tight_layout()
        plt.show()

        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        fig2.suptitle(f"{region} | Divergence Metrics", fontsize=16)

        for i, epoch_name in enumerate(epochs_ordered):
            ax = axes2[i]
            session_matrices = accumulated_data[region][epoch_name]
            if not session_matrices:
                continue

            pop_matrix = np.vstack(session_matrices)
            n_bins = int(pop_matrix.shape[1] / 4)

            # Reshape: (Condition, Time, Neurons)
            reshaped = np.transpose(pop_matrix.reshape(pop_matrix.shape[0], 4, n_bins), (1, 2, 0))

            # Time Axis
            offset = EPOCHS[epoch_name].get("offset", 0)
            t_axis = np.linspace(
                offset - EPOCHS[epoch_name]["t_pre"], offset + EPOCHS[epoch_name]["t_post"], n_bins
            )

            # --- Indices ---
            # 0: L_Cong, 1: L_Incong, 2: R_Cong, 3: R_Incong

            if epoch_name == "Quiescent":
                # Metric: Expectation (Left Block vs Right Block)
                # Left Block = L_Cong(0) + R_Incong(3)
                # Right Block = R_Cong(2) + L_Incong(1)
                vec_L_Block = (reshaped[0] + reshaped[3]) / 2.0
                vec_R_Block = (reshaped[2] + reshaped[1]) / 2.0

                dist = calculate_euclidean_dist(vec_L_Block, vec_R_Block)
                ax.plot(t_axis, dist, color="purple", lw=2, label="Block Bias")
                ax.set_title("Expectation (Block L vs R)")

            else:
                # Metric: Conflict (Congruent vs Incongruent)
                # Left Side: 0 vs 1
                dist_L = calculate_euclidean_dist(reshaped[0], reshaped[1])
                # Right Side: 2 vs 3
                dist_R = calculate_euclidean_dist(reshaped[2], reshaped[3])

                ax.plot(t_axis, dist_L, color="blue", label="Left C vs I")
                ax.plot(t_axis, dist_R, color="red", label="Right C vs I")
                ax.set_title("Conflict (Cong vs Incong)")

            ax.axvline(offset, color="k", linestyle=":")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize="small")

        plt.tight_layout()
        plt.show()


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

    MAX_WORKERS = 8

    print(f"Found {len(task_list)} sessions. Starting extraction with {MAX_WORKERS} cores...")
    t0 = time.time()

    accumulated_data = {reg: {ep: [] for ep in EPOCHS} for reg in MY_REGIONS}

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_single_session,
                pid,
                eid,
                MY_REGIONS,
                EPOCHS,
                USE_SLIDING_WINDOW,
                BIN_SIZE,
                STRIDE,
                BIN_SIZE,
                correct=CORRECT,
            ): pid
            for (pid, eid) in task_list
        }

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                for region, epoch_dict in result.items():
                    for epoch_name, matrix in epoch_dict.items():
                        accumulated_data[region][epoch_name].append(matrix)
            print(f"Progress: {i+1}/{len(task_list)}", end="\r")

    print(f"\nExtraction complete in {time.time() - t0:.2f} seconds.")

    addendum = ""
    if CORRECT:
        addendum += "_correct"
    else:
        addendum += "_incorrect"

    save_path = f"./data/generated/bwm_accumulated_data_{addendum}.pkl"

    print(f"\nSaving data to {save_path}...")
    with open(save_path, "wb") as f:
        pkl.dump(accumulated_data, f)
    print("Save complete.")
