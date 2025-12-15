from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_sample_weight
from tqdm import tqdm
from ibl_info.utils import check_config
import numpy as np

config = check_config()


def run_dual_region_decoder_bootstrapping(
    neural_data_A,
    neural_data_B,
    trial_labels,
    subset_size_D,
    n_bootstraps=50,
    n_splits=5,
    congruent_mask=None,
    incongruent_mask=None,
    scale=True,
):
    """
    Bootstraps linear decoders on subsets of neurons drawn from two different regions/arrays.

    Parameters:
    -----------
    neural_data_A : np.ndarray
        Shape (n_trials, n_neurons_A). Data for Region/Decoder A.
    neural_data_B : np.ndarray
        Shape (n_trials, n_neurons_B). Data for Region/Decoder B.
    trial_labels : np.ndarray
        Shape (1, n) or (n,). Class labels for the trials.
    subset_size_D : int
        Number of neurons to subsample from EACH array (A and B).
    n_bootstraps : int
        Number of resampling iterations.
    n_splits : int
        Number of K-Fold splits.
    congruent_mask : np.ndarray, optional
        Boolean mask for congruent trials.
    incongruent_mask : np.ndarray, optional
        Boolean mask for incongruent trials.
    scale: bool, optional
        If True, applies StandardScaler within the CV loop.

    Returns:
    --------
    list of dicts
        Contains probabilities, metrics, and indices for each bootstrap iteration.
    """

    # 1. Validation and Setup
    X_A = neural_data_A
    X_B = neural_data_B
    y = trial_labels.flatten()

    n_trials_A, n_neurons_A = X_A.shape
    n_trials_B, n_neurons_B = X_B.shape

    # Ensure trial counts match
    if n_trials_A != n_trials_B:
        raise ValueError(f"Trial count mismatch: A has {n_trials_A}, B has {n_trials_B}")

    if len(y) != n_trials_A:
        raise ValueError(f"Labels length {len(y)} does not match trial count {n_trials_A}")

    # Ensure we have enough neurons to sample from
    if subset_size_D > n_neurons_A:
        raise ValueError(f"Cannot select {subset_size_D} neurons from A (size {n_neurons_A})")
    if subset_size_D > n_neurons_B:
        raise ValueError(f"Cannot select {subset_size_D} neurons from B (size {n_neurons_B})")

    # Handle masks
    cong_indices = None
    incong_indices = None

    if congruent_mask is not None:
        congruent_mask = np.array(congruent_mask).flatten().astype(bool)
        cong_indices = np.where(congruent_mask)[0]

    if incongruent_mask is not None:
        incongruent_mask = np.array(incongruent_mask).flatten().astype(bool)
        incong_indices = np.where(incongruent_mask)[0]

    results = []

    for i in range(n_bootstraps):
        # 2. Sub-sample neurons independently from A and B
        # We use permutation[:k] to sample without replacement
        idx_A = np.random.permutation(n_neurons_A)[:subset_size_D]
        idx_B = np.random.permutation(n_neurons_B)[:subset_size_D]

        X_subset_A = X_A[:, idx_A]
        X_subset_B = X_B[:, idx_B]

        # 3. K-Fold Cross Validation
        n_classes = len(np.unique(y))

        # Placeholders (aligned with original trial order)
        probs_A_all = np.zeros((n_trials_A, n_classes))
        probs_B_all = np.zeros((n_trials_A, n_classes))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

        # Note: We split based on y, indices align for both A and B since trials are paired
        for train_idx, test_idx in skf.split(np.zeros(n_trials_A), y):

            # Slice Data
            X_train_A, X_test_A = X_subset_A[train_idx], X_subset_A[test_idx]
            X_train_B, X_test_B = X_subset_B[train_idx], X_subset_B[test_idx]
            y_train = y[train_idx]

            # Scaling
            if scale:
                scaler_A = StandardScaler()
                X_train_A = scaler_A.fit_transform(X_train_A)
                X_test_A = scaler_A.transform(X_test_A)

                scaler_B = StandardScaler()
                X_train_B = scaler_B.fit_transform(X_train_B)
                X_test_B = scaler_B.transform(X_test_B)

            train_weights = compute_sample_weight(class_weight="balanced", y=y_train)

            # Train Decoders
            clf_A = LogisticRegression(solver="lbfgs", max_iter=1000)
            clf_B = LogisticRegression(solver="lbfgs", max_iter=1000)

            clf_A.fit(X_train_A, y_train, sample_weight=train_weights)
            clf_B.fit(X_train_B, y_train, sample_weight=train_weights)

            # Predict
            probs_A_all[test_idx] = clf_A.predict_proba(X_test_A)
            probs_B_all[test_idx] = clf_B.predict_proba(X_test_B)

        # 4. Calculate Metrics
        preds_A = np.argmax(probs_A_all, axis=1)
        preds_B = np.argmax(probs_B_all, axis=1)

        acc_A = accuracy_score(y, preds_A)
        acc_B = accuracy_score(y, preds_B)
        bal_acc_A = balanced_accuracy_score(y, preds_A)
        bal_acc_B = balanced_accuracy_score(y, preds_B)

        metrics_sub = {}

        # Helper for subsets (Congruent/Incongruent)
        def get_subset_metrics(indices, y_full, preds_A_full, preds_B_full, prefix):
            if indices is None or len(indices) == 0:
                return {}
            y_sub = y_full[indices]
            p_A_sub = preds_A_full[indices]
            p_B_sub = preds_B_full[indices]
            return {
                f"accuracy_A_{prefix}": accuracy_score(y_sub, p_A_sub),
                f"accuracy_B_{prefix}": accuracy_score(y_sub, p_B_sub),
                f"balanced_acc_A_{prefix}": balanced_accuracy_score(y_sub, p_A_sub),
                f"balanced_acc_B_{prefix}": balanced_accuracy_score(y_sub, p_B_sub),
            }

        metrics_sub.update(get_subset_metrics(cong_indices, y, preds_A, preds_B, "cong"))
        metrics_sub.update(get_subset_metrics(incong_indices, y, preds_A, preds_B, "incong"))

        # 5. Store Results
        run_data = {
            "iteration": i,
            "probs_A": probs_A_all,
            "probs_B": probs_B_all,
            "probs_A_cong": probs_A_all[cong_indices] if cong_indices is not None else None,
            "probs_A_incong": probs_A_all[incong_indices] if incong_indices is not None else None,
            "probs_B_cong": probs_B_all[cong_indices] if cong_indices is not None else None,
            "probs_B_incong": probs_B_all[incong_indices] if incong_indices is not None else None,
            "accuracy_A": acc_A,
            "accuracy_B": acc_B,
            "balanced_acc_A": bal_acc_A,
            "balanced_acc_B": bal_acc_B,
            "y_true": y,
            "y_cong": y[cong_indices] if cong_indices is not None else None,
            "y_incong": y[incong_indices] if incong_indices is not None else None,
            "neurons_A_indices": idx_A,
            "neurons_B_indices": idx_B,
            "cong_indices": cong_indices,
            "incong_indices": incong_indices,
        }
        run_data.update(metrics_sub)
        results.append(run_data)

    print("Bootstrapping complete.")
    return results
