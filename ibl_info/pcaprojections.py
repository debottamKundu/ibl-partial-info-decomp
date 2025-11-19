import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import math


def analyze_neural_interaction(
    neural_data, labels, n_folds=5, n_bootstraps=50, balance_classes=True, C=1.0
):
    """
    Analyzes the interaction between two complementary halves of the neural population.

    Logic:
    1. Randomly split neurons into two halves (Subset 1 & Subset 2).
    2. Extract PC1 from each half.
    3. Fix PC sign using MEAN CORRELATION (More robust for PC1 than Max Loading).
    4. Fit Logistic Regression: LogOdds ~ b1*PC1_1 + b2*PC1_2 + b3*(PC1_1 * PC1_2)

    Parameters:
    -----------
    neural_data : np.ndarray
        Shape (m, n) where m is total neurons, n is trials/samples.
    labels : np.ndarray
        Shape (1, n) or (n,). The target variable (0 or 1).
    n_folds : int
        Number of cross-validation folds per bootstrap.
    n_bootstraps : int
        Number of times to resample the neuron splits.
    balance_classes : bool
        If True, uses class_weight='balanced'.
    C : float
        Inverse of regularization strength.

    Returns:
    --------
    dict
        Contains mean/std of coefficients, Accuracy, and Explained Variance.
    """

    # 1. Feasibility Checks
    m_neurons, n_trials = neural_data.shape
    labels = labels.reshape(-1)

    if m_neurons < 2:
        raise ValueError(f"Need at least 2 neurons to split into subsets. Found {m_neurons}.")

    split_point = m_neurons // 2
    max_splits = math.comb(m_neurons, split_point)

    if n_bootstraps > max_splits:
        print(
            f"(!) N={m_neurons} allows only {max_splits} unique splits. Reducing bootstraps {n_bootstraps}->{max_splits}."
        )
        n_bootstraps = max_splits

    split_point = m_neurons // 2

    # Storage for results
    bootstrap_coefs = []
    bootstrap_scores = []
    bootstrap_vars = []

    X_transposed = neural_data.T
    all_indices = np.arange(m_neurons)
    cw_param = "balanced" if balance_classes else None

    # --- Bootstrap Loop ---
    for b in range(n_bootstraps):
        # Randomly shuffle and split into two halves
        np.random.shuffle(all_indices)
        idx_subset_1 = all_indices[:split_point]
        idx_subset_2 = all_indices[split_point:]

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)

        fold_coefs = []
        fold_scores = []
        fold_vars = []

        for train_index, test_index in kf.split(X_transposed):
            X_train, X_test = X_transposed[train_index], X_transposed[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            X_sub1_train = X_train[:, idx_subset_1]
            X_sub1_test = X_test[:, idx_subset_1]
            X_sub2_train = X_train[:, idx_subset_2]
            X_sub2_test = X_test[:, idx_subset_2]

            # --- PCA Step (Single Component) ---
            pca1 = PCA(n_components=1)
            pca2 = PCA(n_components=1)

            # Shape: (n_samples, 1)
            pc1_train = pca1.fit_transform(X_sub1_train)
            pc2_train = pca2.fit_transform(X_sub2_train)

            # Track Variance
            fold_vars.append(
                [pca1.explained_variance_ratio_[0], pca2.explained_variance_ratio_[0]]
            )

            pc1_test = pca1.transform(X_sub1_test)
            pc2_test = pca2.transform(X_sub2_test)

            # --- Sign Ambiguity Check (Reverted to Mean Correlation) ---
            # This anchors "Positive PC" to "High Average Firing Rate"

            # Check Subset 1
            corr1 = np.corrcoef(pc1_train.flatten(), y_train)[0, 1]
            if corr1 < 0:
                pc1_train *= -1
                pc1_test *= -1

            # Check Subset 2
            corr2 = np.corrcoef(pc2_train.flatten(), y_train)[0, 1]
            if corr2 < 0:
                pc2_train *= -1
                pc2_test *= -1

            # --- Interaction Term ---
            inter_train = pc1_train * pc2_train
            inter_test = pc1_test * pc2_test

            # --- Stack 3 Features ---
            features_train = np.hstack([pc1_train, pc2_train, inter_train])
            features_test = np.hstack([pc1_test, pc2_test, inter_test])

            # --- Standardization ---
            scaler = StandardScaler()
            features_train = scaler.fit_transform(features_train)
            features_test = scaler.transform(features_test)

            # --- Logistic Regression ---
            model = LogisticRegression(solver="lbfgs", C=C, class_weight=cw_param, max_iter=1000)
            model.fit(features_train, y_train)

            fold_coefs.append(model.coef_[0])
            fold_scores.append(model.score(features_test, y_test))

        bootstrap_coefs.append(np.mean(fold_coefs, axis=0))
        bootstrap_scores.append(np.mean(fold_scores))
        bootstrap_vars.append(np.mean(fold_vars, axis=0))

        if (b + 1) % 10 == 0:
            print(f"Completed bootstrap {b + 1}/{n_bootstraps}")

    # 4. Aggregate Results
    bootstrap_coefs = np.array(bootstrap_coefs)  # Shape (n_bootstraps, 3)
    bootstrap_scores = np.array(bootstrap_scores)
    bootstrap_vars = np.array(bootstrap_vars)  # Shape (n_bootstraps, 2)

    means = np.mean(bootstrap_coefs, axis=0)
    stds = np.std(bootstrap_coefs, axis=0)

    results = {
        "mean_coefficients": {
            "PC1_Subset1": means[0],
            "PC1_Subset2": means[1],
            "Interaction": means[2],
        },
        "std_coefficients": {
            "PC1_Subset1": stds[0],
            "PC1_Subset2": stds[1],
            "Interaction": stds[2],
        },
        "mean_test_accuracy": np.mean(bootstrap_scores),
        "std_test_accuracy": np.std(bootstrap_scores),
        "mean_explained_variance": {
            "Subset1": np.mean(bootstrap_vars[:, 0]),
            "Subset2": np.mean(bootstrap_vars[:, 1]),
        },
        "all_bootstrap_coefficients": bootstrap_coefs,
        "all_bootstrap_accuracy": bootstrap_scores,
        "all_bootstrap_explained_variance": bootstrap_vars,
    }

    return results


def analyze_inter_region_interaction(
    region1_data, region2_data, labels, n_folds=5, balance_classes=True, C=1.0
):
    """
    Analyzes the interaction between two specific brain regions in predicting BINARY labels.
    Uses PC1 of Region 1 and PC1 of Region 2 + Interaction Term.

    Parameters:
    -----------
    region1_data : np.ndarray
        Shape (m1, n) - Neurons from Region 1.
    region2_data : np.ndarray
        Shape (m2, n) - Neurons from Region 2.
    labels : np.ndarray
        Shape (1, n) or (n,). The target variable (0 or 1).
    n_folds : int
        Number of cross-validation folds.
    balance_classes : bool
        If True, uses class_weight='balanced'.
    C : float
        Inverse of regularization strength.

    Returns:
    --------
    dict
        Contains mean/std of coefficients and Accuracy across CV folds.
    """

    # 1. Feasibility Checks
    m1, n_trials1 = region1_data.shape
    m2, n_trials2 = region2_data.shape

    labels = labels.reshape(-1)
    if labels.shape[0] != n_trials1:
        raise ValueError(
            f"Label count ({labels.shape[0]}) does not match Region 1 trials ({n_trials1})."
        )
    if n_trials1 != n_trials2:
        raise ValueError(f"Trial mismatch: Region 1 has {n_trials1}, Region 2 has {n_trials2}.")

    print(f"--- Starting Inter-Region Analysis ---")
    print(f"Region 1 Neurons: {m1}, Region 2 Neurons: {m2}, Trials: {n_trials1}")

    # Transpose for sklearn (n_samples, n_features)
    X1_T = region1_data.T
    X2_T = region2_data.T

    # Prepare Cross-Validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_coefs = []
    fold_scores = []

    cw_param = "balanced" if balance_classes else None

    fold_idx = 1

    # We iterate through indices since we have two separate data matrices
    for train_index, test_index in kf.split(X1_T):

        # --- Split Data ---
        X1_train, X1_test = X1_T[train_index], X1_T[test_index]
        X2_train, X2_test = X2_T[train_index], X2_T[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # --- PCA Step (Separately for each region) ---
        pca1 = PCA(n_components=1)
        pca2 = PCA(n_components=1)

        # Region 1 PCA
        pc1_train = pca1.fit_transform(X1_train)
        pc1_test = pca1.transform(X1_test)

        # Region 2 PCA
        pc2_train = pca2.fit_transform(X2_train)
        pc2_test = pca2.transform(X2_test)

        # --- Sign Ambiguity Check (CRITICAL) ---
        # Ensure PC aligns with mean activity of its own region
        mean_act1 = np.mean(X1_train, axis=1).reshape(-1, 1)
        if np.corrcoef(pc1_train.T, mean_act1.T)[0, 1] < 0:
            pc1_train *= -1
            pc1_test *= -1

        mean_act2 = np.mean(X2_train, axis=1).reshape(-1, 1)
        if np.corrcoef(pc2_train.T, mean_act2.T)[0, 1] < 0:
            pc2_train *= -1
            pc2_test *= -1

        # --- Interaction Term ---
        inter_train = pc1_train * pc2_train
        inter_test = pc1_test * pc2_test

        # --- Stack Features ---
        # [Region1_PC, Region2_PC, Interaction]
        features_train = np.hstack([pc1_train, pc2_train, inter_train])
        features_test = np.hstack([pc1_test, pc2_test, inter_test])

        # --- Standardization ---
        scaler = StandardScaler()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

        # --- Logistic Regression ---
        model = LogisticRegression(solver="lbfgs", C=C, class_weight=cw_param)
        model.fit(features_train, y_train)

        fold_coefs.append(model.coef_[0])
        fold_scores.append(model.score(features_test, y_test))

        fold_idx += 1

    # 4. Aggregate Results (Across Folds)
    fold_coefs = np.array(fold_coefs)
    fold_scores = np.array(fold_scores)

    results = {
        "mean_coefficients": {
            "Region1_PC1": np.mean(fold_coefs[:, 0]),
            "Region2_PC1": np.mean(fold_coefs[:, 1]),
            "Interaction": np.mean(fold_coefs[:, 2]),
        },
        "std_coefficients": {
            "Region1_PC1": np.std(fold_coefs[:, 0]),
            "Region2_PC1": np.std(fold_coefs[:, 1]),
            "Interaction": np.std(fold_coefs[:, 2]),
        },
        "mean_test_accuracy": np.mean(fold_scores),
        "std_test_accuracy": np.std(fold_scores),
        "raw_fold_coefficients": fold_coefs,
        "raw_fold_accuracy": fold_scores,
    }

    return results
