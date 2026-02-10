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
MY_REGIONS = config["stim_prior_regions"]
MIN_NEURONS = config["min_units"]
BIN_SIZE = 0.01  # 10ms
STRIDE = 0.001  # 1ms
USE_SLIDING_WINDOW = config["use_sliding_window"]
MIN_TRIALS = 1  # Minimum trials per condition to include session

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

COND_NAMES = [
    "L_Cong_Corr",
    "L_Cong_Err",
    "L_Incong_Corr",
    "L_Incong_Err",
    "R_Cong_Corr",
    "R_Cong_Err",
    "R_Incong_Corr",
    "R_Incong_Err",
]

# Base colors for the 4 types (Dark Blue, Light Blue, Dark Red, Light Red)
# We will use Solid Lines for Correct, Dashed for Error
BASE_COLORS = ["#00008B", "#6495ED", "#8B0000", "#FA8072"]


def get_trial_masks(trials, simple=False):
    """
    Returns boolean masks for 8 conditions (Correct & Error).
    """
    masks = {}

    has_contrast_L = ~np.isnan(trials["contrastLeft"])
    has_contrast_R = ~np.isnan(trials["contrastRight"])

    is_L_block = trials["probabilityLeft"] == 0.8
    is_R_block = trials["probabilityLeft"] == 0.2

    is_correct = trials["feedbackType"] == 1
    is_error = trials["feedbackType"] == -1

    # Left Block Conditions
    masks["L_Cong_Corr"] = has_contrast_L & is_L_block & is_correct
    masks["L_Cong_Err"] = has_contrast_L & is_L_block & is_error
    masks["R_Incong_Corr"] = has_contrast_R & is_L_block & is_correct
    masks["R_Incong_Err"] = has_contrast_R & is_L_block & is_error

    # Right Block Conditions
    masks["R_Cong_Corr"] = has_contrast_R & is_R_block & is_correct
    masks["R_Cong_Err"] = has_contrast_R & is_R_block & is_error
    masks["L_Incong_Corr"] = has_contrast_L & is_R_block & is_correct
    masks["L_Incong_Err"] = has_contrast_L & is_R_block & is_error

    if simple:
        masks["Correct"] = is_correct
        masks["Error"] = is_error

    return masks


def process_single_session(
    pid,
    eid,
    requested_regions,
    epochs_config,
    use_slide,
    win_size,
    stride,
    bin_simple,
    simple_mask=False,
):
    """
    Loads one session, extracts spikes, and computes PETHs for 8 conditions.
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

        # Get masks for all 8 conditions
        masks = get_trial_masks(trials, simple_mask)

        if simple_mask:
            COND_NAMES = ["Correct", "Error"]

        for cond in COND_NAMES:
            if np.sum(masks[cond]) < MIN_TRIALS:
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

                # Stack: (NeuronsxTime * 8_Conditions)
                session_results[region][epoch_name] = np.hstack(epoch_stack)

        return session_results

    except Exception as e:
        print(f"Error in {eid}: {e}")
        return None


def plot_pcas_and_euclids(accumulated_data):
    for region in MY_REGIONS:
        print(f"\n--- Visualizing Region: {region} ---")

        epochs_ordered = ["Quiescent", "Stimulus", "Choice"]

        # --- FIG 1: PCA State Space ---
        fig1 = plt.figure(figsize=(15, 5))
        fig1.suptitle(f"{region} | PCA Common Space (Correct + Error)", fontsize=16)

        for i, epoch_name in enumerate(epochs_ordered):
            session_matrices = accumulated_data[region][epoch_name]
            if not session_matrices:
                continue

            # Shape: (Neurons, Total_Time) - Stacked across sessions
            pop_matrix = np.vstack(session_matrices)

            # PCA on ALL 8 conditions together
            pca = PCA(n_components=3)
            X_embedded = pca.fit_transform(pop_matrix.T)  # (Total_Time, 3)

            n_bins = int(X_embedded.shape[0] / 8)  # Divided by 8 conditions
            trajs = X_embedded.reshape(8, n_bins, 3)

            ax = fig1.add_subplot(1, 3, i + 1, projection="3d")

            # Plot the 8 Conditions
            # Mapping 8 conditions to 4 Base Colors
            # 0,1 -> Base0 (L_Cong)
            # 2,3 -> Base1 (L_Incong) ...

            for c in range(8):
                color_idx = c // 2  # 0 or 1 -> 0; 2 or 3 -> 1, etc.
                is_err = (c % 2) != 0  # Even indices are Corr, Odd are Err

                style = "--" if is_err else "-"
                alpha = 0.6 if is_err else 1.0
                lw = 1.5 if is_err else 2.5

                ax.plot(
                    trajs[c, :, 0],
                    trajs[c, :, 1],
                    trajs[c, :, 2],
                    color=BASE_COLORS[color_idx],
                    linestyle=style,
                    label=COND_NAMES[c] if i == 0 else "",  # Legend only on first plot or specific
                    lw=lw,
                    alpha=alpha,
                )
                # Mark start
                ax.scatter(
                    trajs[c, 0, 0],
                    trajs[c, 0, 1],
                    trajs[c, 0, 2],
                    color=BASE_COLORS[color_idx],
                    s=10,
                )

            ax.set_title(epoch_name)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            if i == 0:
                ax.legend(loc="upper left", fontsize="xx-small", frameon=False)

        plt.tight_layout()
        plt.show()

        # --- FIG 2: 8x8 Distance Matrices ---
        fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig2.suptitle(f"{region} | Representational Distance (8x8)", fontsize=16)

        if len(epochs_ordered) == 1:
            axes = [axes]

        for i, epoch_name in enumerate(epochs_ordered):
            ax = axes[i]
            session_matrices = accumulated_data[region][epoch_name]

            if not session_matrices:
                ax.text(0.5, 0.5, "No Data", ha="center")
                continue

            pop_matrix = np.vstack(session_matrices)
            n_bins = int(pop_matrix.shape[1] / 8)

            # Reshape: (Condition, Time, Neurons) -> (8, Time, Neurons)
            reshaped = np.transpose(pop_matrix.reshape(pop_matrix.shape[0], 8, n_bins), (1, 2, 0))

            dist_matrix = np.zeros((8, 8))

            for r in range(8):
                for c in range(8):
                    if r == c:
                        dist_matrix[r, c] = 0
                    else:
                        dists_over_time = np.linalg.norm(reshaped[r] - reshaped[c], axis=1)
                        dist_matrix[r, c] = np.mean(dists_over_time)

            im = ax.imshow(
                dist_matrix, cmap="magma", origin="upper"
            )  # 'magma' often good for dists

            # Add text (optional, might be crowded for 8x8)
            for r in range(8):
                for c in range(8):
                    val = dist_matrix[r, c]
                    # Only show text if meaningful size, else it's clutter
                    ax.text(
                        c, r, f"{val:.1f}", ha="center", va="center", color="white", fontsize=7
                    )

            ax.set_title(epoch_name)
            ax.set_xticks(np.arange(8))
            ax.set_yticks(np.arange(8))
            ax.set_xticklabels(COND_NAMES, rotation=90, fontsize=8)
            ax.set_yticklabels(COND_NAMES, fontsize=8)

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
    simple_mask = config["simple_mask"]

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
                simple_mask,
            ): pid
            for (pid, eid) in task_list
        }

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                for region, epoch_dict in result.items():
                    for epoch_name, matrix in epoch_dict.items():
                        # get all animals
                        accumulated_data[region][epoch_name].append(matrix)
            print(f"Progress: {i+1}/{len(task_list)}", end="\r")

    print(f"\nExtraction complete in {time.time() - t0:.2f} seconds.")

    save_path = f"./data/generated/bwm_accumulated_data_correct_incorrect.pkl"

    print(f"\nSaving data to {save_path}...")
    with open(save_path, "wb") as f:
        pkl.dump(accumulated_data, f)
    print("Save complete.")
