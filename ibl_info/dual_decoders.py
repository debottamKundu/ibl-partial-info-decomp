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


def compute_four_group_weights(y, congruent_mask):

    # Create the 4 groups
    # y is assumed to be 0 or 1. If -1/1
    # We combine y and mask to get unique group IDs: 0, 1, 2, 3

    # 0: Left (0) & Incongruent (False)
    # 1: Left (0) & Congruent (True)
    # 2: Right (1) & Incongruent (False)
    # 3: Right (1) & Congruent (True)

    group_labels = y * 2 + congruent_mask.astype(int)

    # Count occurrences of each group
    unique_groups, counts = np.unique(group_labels, return_counts=True)
    n_samples = len(y)
    n_groups = len(unique_groups)

    # Calculate weight for each group: N_total / (N_groups * N_group_i)
    # This ensures each group contributes equally to the loss.
    weights = np.zeros(n_samples)
    for group, count in zip(unique_groups, counts):
        weight = n_samples / (n_groups * count)
        weights[group_labels == group] = weight

    return weights


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

            if congruent_mask is not None:
                mask_train = congruent_mask[train_idx]

            # Scaling
            if scale:
                scaler_A = StandardScaler()
                X_train_A = scaler_A.fit_transform(X_train_A)
                X_test_A = scaler_A.transform(X_test_A)

                scaler_B = StandardScaler()
                X_train_B = scaler_B.fit_transform(X_train_B)
                X_test_B = scaler_B.transform(X_test_B)

            if congruent_mask is None:
                train_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            else:
                train_weights = compute_four_group_weights(y_train, mask_train)

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


def compute_null_distribution(
    neural_data_A,
    neural_data_B,
    trial_labels,
    subset_size_D,
    n_permutations=50,  # Standard is usually 1000 or 10000
    n_splits=5,
    scale=True,
):
    """
    Generates a null distribution of accuracies by shuffling trial labels.
    """

    # Setup
    X_A = neural_data_A
    X_B = neural_data_B
    y = trial_labels.flatten()
    n_trials, n_neurons_A = X_A.shape
    _, n_neurons_B = X_B.shape

    null_metrics = {"acc_A": [], "acc_B": []}

    print(f"Generating Null Distribution ({n_permutations} permutations)...")

    for i in range(n_permutations):
        # 1. Shuffle Labels (The Crucial Step)
        # We shuffle y once per permutation so the decoder learns noise
        y_shuffled = np.random.permutation(y)

        # 2. Subsample Neurons (To match the variance of your real run)
        idx_A = np.random.permutation(n_neurons_A)[:subset_size_D]
        idx_B = np.random.permutation(n_neurons_B)[:subset_size_D]

        X_sub_A = X_A[:, idx_A]
        X_sub_B = X_B[:, idx_B]

        # 3. CV Loop (Fast version)
        # We collect predictions across folds to get one accuracy score per permutation
        preds_A_all = np.zeros(n_trials)
        preds_B_all = np.zeros(n_trials)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

        # Note: Split based on the SHUFFLED labels to maintain class balance in folds
        for train_idx, test_idx in skf.split(np.zeros(n_trials), y_shuffled):

            # Slice
            X_train_A, X_test_A = X_sub_A[train_idx], X_sub_A[test_idx]
            X_train_B, X_test_B = X_sub_B[train_idx], X_sub_B[test_idx]
            y_train = y_shuffled[train_idx]

            # Scale
            if scale:
                scaler_A = StandardScaler().fit(X_train_A)
                X_train_A = scaler_A.transform(X_train_A)
                X_test_A = scaler_A.transform(X_test_A)

                scaler_B = StandardScaler().fit(X_train_B)
                X_train_B = scaler_B.transform(X_train_B)
                X_test_B = scaler_B.transform(X_test_B)

            # Fit (Lightweight settings for speed)
            clf_A = LogisticRegression(solver="lbfgs", max_iter=200, class_weight="balanced")
            clf_B = LogisticRegression(solver="lbfgs", max_iter=200, class_weight="balanced")

            clf_A.fit(X_train_A, y_train)
            clf_B.fit(X_train_B, y_train)

            # Predict
            preds_A_all[test_idx] = clf_A.predict(X_test_A)
            preds_B_all[test_idx] = clf_B.predict(X_test_B)

        # 4. Store Score
        null_metrics["acc_A"].append(accuracy_score(y_shuffled, preds_A_all))
        null_metrics["acc_B"].append(accuracy_score(y_shuffled, preds_B_all))

    return null_metrics


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


def run_dual_region_decoder_bootstrapping_hyperparamopt(
    neural_data_A,
    neural_data_B,
    trial_labels,
    subset_size_D,
    n_bootstraps=50,
    n_splits=5,
    congruent_mask=None,
    incongruent_mask=None,
    scale=True,
    param_grid=None,  # <--- NEW: Param grid for optimization
):
    """
    Bootstraps linear decoders on subsets of neurons from two regions (A and B)
    using Nested Cross-Validation to optimize hyperparameters.
    """

    # 1. Validation and Setup
    if param_grid is None:
        # Default optimization grid for Logistic Regression regularization
        param_grid = {
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "scaler": [StandardScaler(), "passthrough"],
        }

    X_A = neural_data_A
    X_B = neural_data_B
    y = trial_labels.flatten()

    n_trials_A, n_neurons_A = X_A.shape
    n_trials_B, n_neurons_B = X_B.shape

    # Validation checks
    if n_trials_A != n_trials_B:
        raise ValueError(f"Trial count mismatch: A has {n_trials_A}, B has {n_trials_B}")
    if len(y) != n_trials_A:
        raise ValueError(f"Labels length {len(y)} does not match trial count {n_trials_A}")
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
    print(f"Starting Dual-Region Bootstrapping ({n_bootstraps} runs) with Nested CV...")

    for i in range(n_bootstraps):
        # 2. Sub-sample neurons independently from A and B
        idx_A = np.random.permutation(n_neurons_A)[:subset_size_D]
        idx_B = np.random.permutation(n_neurons_B)[:subset_size_D]

        X_subset_A = X_A[:, idx_A]
        X_subset_B = X_B[:, idx_B]

        # 3. K-Fold Cross Validation
        n_classes = len(np.unique(y))

        # Placeholders for probability outputs
        probs_A_all = np.zeros((n_trials_A, n_classes))
        probs_B_all = np.zeros((n_trials_A, n_classes))

        # Track best params
        best_params_A = []
        best_params_B = []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        for train_idx, test_idx in skf.split(np.zeros(n_trials_A), y):

            # Slice Data
            X_train_A, X_test_A = X_subset_A[train_idx], X_subset_A[test_idx]
            X_train_B, X_test_B = X_subset_B[train_idx], X_subset_B[test_idx]
            y_train = y[train_idx]

            # Determine weights
            if congruent_mask is None:
                train_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            else:
                # Assuming 'compute_four_group_weights' is available in your scope
                # If not, fallback to standard balanced weights
                mask_train = congruent_mask[train_idx]
                try:
                    train_weights = compute_four_group_weights(y_train, mask_train)
                except NameError:
                    train_weights = compute_sample_weight(class_weight="balanced", y=y_train)

            # --- Define Pipelines ---
            # We put the estimator in a pipeline step named 'clf' to target it in param_grid
            steps_A = [("clf", LogisticRegression(solver="liblinear", max_iter=1000))]
            steps_B = [("clf", LogisticRegression(solver="liblinear", max_iter=1000))]

            # if scale:
            #     steps_A.insert(0, ("scaler", StandardScaler()))  # type: ignore
            #     steps_B.insert(0, ("scaler", StandardScaler()))  # type: ignore

            # pipeline_A = Pipeline(steps_A)
            # pipeline_B = Pipeline(steps_B)

            pipeline_A = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(solver="liblinear", max_iter=1000)),
                ]
            )

            pipeline_B = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(solver="liblinear", max_iter=1000)),
                ]
            )

            # --- Inner CV (Grid Search) ---
            # Optimize Region A
            grid_A = GridSearchCV(
                pipeline_A, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1
            )
            grid_A.fit(X_train_A, y_train, clf__sample_weight=train_weights)

            # Optimize Region B
            grid_B = GridSearchCV(
                pipeline_B, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1
            )
            grid_B.fit(X_train_B, y_train, clf__sample_weight=train_weights)

            # Store Best Params
            best_params_A.append(grid_A.best_params_["clf__C"])
            best_params_B.append(grid_B.best_params_["clf__C"])

            # Predict (using best model automatically)
            probs_A_all[test_idx] = grid_A.predict_proba(X_test_A)
            probs_B_all[test_idx] = grid_B.predict_proba(X_test_B)

        # 4. Calculate Metrics
        preds_A = np.argmax(probs_A_all, axis=1)
        preds_B = np.argmax(probs_B_all, axis=1)

        acc_A = accuracy_score(y, preds_A)
        acc_B = accuracy_score(y, preds_B)
        bal_acc_A = balanced_accuracy_score(y, preds_A)
        bal_acc_B = balanced_accuracy_score(y, preds_B)

        metrics_sub = {}

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
            "best_params_A": best_params_A,
            "best_params_B": best_params_B,
            "accuracy_A": acc_A,
            "accuracy_B": acc_B,
            "balanced_acc_A": bal_acc_A,
            "balanced_acc_B": bal_acc_B,
            "probs_A_cong": probs_A_all[cong_indices] if cong_indices is not None else None,
            "probs_A_incong": probs_A_all[incong_indices] if incong_indices is not None else None,
            "probs_B_cong": probs_B_all[cong_indices] if cong_indices is not None else None,
            "probs_B_incong": probs_B_all[incong_indices] if incong_indices is not None else None,
            "y_true": y,
            "neurons_A_indices": idx_A,
            "neurons_B_indices": idx_B,
        }
        run_data.update(metrics_sub)
        results.append(run_data)

    print("Bootstrapping complete.")
    return results
