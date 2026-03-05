import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import neo
import quantities as pq
from elephant.gpfa import GPFA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def orthogonalize_gpfa_with_pca(trajectories):
    """
    Applies PCA on top of GPFA trajectories to ensure the dimensions
    are strictly ordered by the amount of variance they explain.

    Parameters:
    - trajectories: numpy array of shape (N_trials, latent_dim, N_timebins)

    Returns:
    - pca_trajectories: transformed array of the same shape
    - explained_variance: list showing the % of variance explained by each new dimension
    """
    n_trials, latent_dim, n_timebins = trajectories.shape

    # (Trials, Time, Dim)
    flattened = np.transpose(trajectories, (0, 2, 1)).reshape(-1, latent_dim)
    pca = PCA(n_components=latent_dim)
    flattened_pca = pca.fit_transform(flattened)

    pca_trajectories = flattened_pca.reshape(n_trials, n_timebins, latent_dim).transpose(0, 2, 1)

    variance_ratio = pca.explained_variance_ratio_ * 100
    print("Variance explained by new orthogonal dimensions:")
    for i, var in enumerate(variance_ratio):
        print(f"  Dim {i+1}: {var:.2f}%")

    return pca_trajectories, variance_ratio


def plot_gpfa_trajectories(trajectories, condition_masks, bin_size_ms=20, time_before_stim=500):
    """
    Plots the trial-averaged GPFA trajectories for each condition in 2D space.

    Parameters:
    - trajectories: numpy array (N_trials, latent_dim, N_timebins) from GPFA output
    - condition_masks: dict of boolean masks for your conditions
    - bin_size_ms: time duration of each bin
    - time_before_stim: total length of the quiescent period in ms
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    n_timebins = trajectories.shape[2]
    time_vector = np.linspace(-time_before_stim, 0, n_timebins)

    colors = plt.cm.tab10(np.linspace(0, 1, len(condition_masks)))  # type: ignore

    for i, (cond_name, mask) in enumerate(condition_masks.items()):

        cond_trials = trajectories[mask]

        mean_trajectory = np.nanmean(cond_trials, axis=0)  # Shape: (latent_dim, N_timebins)

        x_dim = mean_trajectory[0, :]
        y_dim = mean_trajectory[1, :]

        ax.plot(x_dim, y_dim, label=cond_name, color=colors[i], linewidth=2.5)

        ax.scatter(x_dim[0], y_dim[0], color=colors[i], marker="o", s=50)

        ax.scatter(x_dim[-1], y_dim[-1], color=colors[i], marker="X", s=150, edgecolor="black")

    ax.set_title("GPFA Latent State Trajectories", fontsize=14)
    ax.set_xlabel("GPFA Latent Dimension 1", fontsize=12)
    ax.set_ylabel("GPFA Latent Dimension 2", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def compute_mmd(X, Y, gamma=None):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples X and Y
    using a Gaussian (RBF) kernel.

    Parameters:
    - X: array-like, shape (n_trials_A, n_features)
    - Y: array-like, shape (n_trials_B, n_features)
    - gamma: Kernel coefficient for RBF. If None, defaults to 1 / n_features
    """

    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)

    mmd_squared = XX.mean() + YY.mean() - 2 * XY.mean()
    return np.sqrt(np.maximum(mmd_squared, 0))


def extract_pre_stimulus_state(trajectories, condition_masks):
    """
    Extracts the final point of the GPFA trajectory (the exact pre-stimulus state)
    for single-trial analysis (like Logistic Regression or MMD).
    """

    final_timebin_idx = -1

    pre_stim_states = trajectories[:, :, final_timebin_idx]

    states_by_condition = {}
    for cond_name, mask in condition_masks.items():
        states_by_condition[cond_name] = pre_stim_states[mask]

    return pre_stim_states, states_by_condition


def run_gpfa_and_mmd(spiketrains, condition_masks, latent_dim=8, bin_size_s=0.05, onlygpfa=False):
    """
    Fits GPFA directly on neo.SpikeTrain objects and computes the MMD distance matrix.

    Parameters:
    - spiketrains: list of lists of neo.SpikeTrain (output from get_spiketrains_for_elephant)
    - condition_masks: dict mapping condition names to boolean arrays of length N_trials
    - latent_dim: Number of latent dimensions for GPFA
    - bin_size_s: The bin size in seconds (e.g., 0.05 for 50ms)

    Returns:
    - mmd_matrix: (8x8) numpy array of distribution distances
    - condition_names: list of the 8 condition names matching the matrix rows/cols
    - trajectories: The extracted GPFA latent trajectories (N_trials, latent_dim, N_timebins)
    """

    print(f"Fitting GPFA with {latent_dim} latent dimensions...")
    gpfa = GPFA(bin_size=bin_size_s * pq.s, x_dim=latent_dim)

    latent_trajectories = gpfa.fit_transform(spiketrains)

    trajectories = np.array(latent_trajectories)

    if onlygpfa:
        return None, None, trajectories

    trajectories = np.stack(trajectories)  # type: ignore

    print("Computing MMD Distance Matrix...")
    n_trials = trajectories.shape[0]
    condition_names = list(condition_masks.keys())
    n_conditions = len(condition_names)
    mmd_matrix = np.zeros((n_conditions, n_conditions))

    flattened_trajectories = trajectories.reshape(n_trials, -1)

    for i, cond_A in enumerate(condition_names):
        for j, cond_B in enumerate(condition_names):
            if i > j:
                # The matrix is symmetric
                mmd_matrix[i, j] = mmd_matrix[j, i]
                continue
            elif i == j:
                # Distance to itself is 0
                mmd_matrix[i, j] = 0.0
                continue

            # Extract the trial clouds for both conditions
            mask_A = condition_masks[cond_A]
            mask_B = condition_masks[cond_B]

            # Optional safety check: skip if a condition has 0 trials
            if not np.any(mask_A) or not np.any(mask_B):
                mmd_matrix[i, j] = np.nan
                continue

            cloud_A = flattened_trajectories[mask_A]
            cloud_B = flattened_trajectories[mask_B]

            # Compute MMD
            dist = compute_mmd(cloud_A, cloud_B)
            mmd_matrix[i, j] = dist

    print("Done!")
    return mmd_matrix, condition_names, trajectories


def get_spiketrains_for_elephant(
    spike_times,
    spike_clusters,
    cluster_ids,
    align_times,
    pre_time=0.4,
    post_time=1,
    bin_size=0.01,
    weights=None,
):
    """
    Event aligned SpikeTrains for multiple clusters (Formatted for Elephant GPFA)
    :param spike_times: Array of all spike times in the session (seconds)
    :param spike_clusters: Array of cluster IDs corresponding to spike_times
    :param cluster_ids: Array of cluster IDs to extract
    :param align_times: Array of event times to align trials to
    :param pre_time: Window before align_time (seconds)
    :param post_time: Window after align_time (seconds)
    :param bin_size: Only used here to compute tscale to maintain return signature
    :param weights: Ignored in this version since GPFA works on raw spikes
    :return: spiketrains (list of lists of neo.SpikeTrain), tscale
    """

    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    ts = np.repeat(align_times[:, np.newaxis], tscale.size, axis=1) + tscale
    epoch_idxs = np.searchsorted(spike_times, np.c_[ts[:, 0], ts[:, -1]])

    spiketrains = []

    for i, (ep, t) in enumerate(zip(epoch_idxs, ts)):

        trial_spikes_abs = spike_times[ep[0] : ep[1]]
        trial_clusters = spike_clusters[ep[0] : ep[1]]

        align_t = align_times[i]
        trial_spikes_rel = trial_spikes_abs - align_t

        trial_trains = []

        for cid in cluster_ids:

            neuron_spikes = trial_spikes_rel[trial_clusters == cid]

            st = neo.SpikeTrain(
                neuron_spikes * pq.s, t_start=-pre_time * pq.s, t_stop=post_time * pq.s
            )
            trial_trains.append(st)

        spiketrains.append(trial_trains)
    tscale = (tscale[:-1] + tscale[1:]) / 2

    return spiketrains, tscale


from mpl_toolkits.mplot3d import Axes3D


def plot_gpfa_3d_trajectories(trajectories, condition_masks, dims=(0, 1, 2)):
    """
    Plots the trial-averaged GPFA trajectories for each condition in 3D space.

    Parameters:
    - trajectories: numpy array (N_trials, latent_dim, N_timebins) from GPFA
    - condition_masks: dict of boolean masks for your conditions
    - dims: tuple of the 3 latent dimensions to plot (default is the first three)
    """

    if trajectories.shape[1] < 3:
        raise ValueError("Cannot plot 3D trajectories: Latent dimension is less than 3.")

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.linspace(0, 1, len(condition_masks)))

    for i, (cond_name, mask) in enumerate(condition_masks.items()):

        cond_trials = trajectories[mask]

        if len(cond_trials) == 0:
            continue

        mean_trajectory = np.nanmean(cond_trials, axis=0)  # Shape: (latent_dim, N_timebins)

        x_dim = mean_trajectory[dims[0], :]
        y_dim = mean_trajectory[dims[1], :]
        z_dim = mean_trajectory[dims[2], :]

        ax.plot(x_dim, y_dim, z_dim, label=cond_name, color=colors[i], linewidth=2.5)

        ax.scatter(x_dim[0], y_dim[0], z_dim[0], color=colors[i], marker="o", s=50)
        ax.scatter(
            x_dim[-1], y_dim[-1], z_dim[-1], color=colors[i], marker="X", s=150, edgecolor="black"
        )

    ax.set_title("3D GPFA Neural State Trajectories", fontsize=16)
    ax.set_xlabel(f"Latent Dim {dims[0] + 1}", labelpad=10)
    ax.set_ylabel(f"Latent Dim {dims[1] + 1}", labelpad=10)
    ax.set_zlabel(f"Latent Dim {dims[2] + 1}", labelpad=10)

    ax.legend(bbox_to_anchor=(1.15, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def decode_from_final_state(trajectories, labels, outer_folds=5, inner_folds=3):
    """
    Trains a cross-validated decoder on the final timepoint of the GPFA trajectories.

    Parameters:
    - trajectories: (N_trials, latent_dim, N_timebins) array from GPFA
    - labels: (N_trials,) array of binary labels (e.g., 0 for Low Rew, 1 for High Rew)
    - cv_folds: Number of cross-validation splits

    Returns:
    - mean_accuracy: Float representing how well the latent state predicts the label
    """

    final_quiescent_states = trajectories[:, :, -1]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000),
            ),
        ]
    )

    param_grid = {"logreg__C": [0.001, 0.01, 0.1, 1.0, 10.0]}

    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)

    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    nested_scores = cross_val_score(
        estimator=clf,
        X=final_quiescent_states,
        y=labels,
        cv=outer_cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    return np.mean(nested_scores)
