import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore


def ideal_rsa_matrices():
    """
    Constructs 8x8 Model RDMs (Representational Dissimilarity Matrices).
    Returns a dictionary of flattened model vectors (upper triangle) to use as predictors.
    
    Indices (8 conditions):
    0: L_Cong_Corr  (Stim L, Block L, Move L)
    1: L_Cong_Err   (Stim L, Block L, Move R)
    2: L_Inc_Corr   (Stim L, Block R, Move L)
    3: L_Inc_Err    (Stim L, Block R, Move R)
    4: R_Cong_Corr  (Stim R, Block R, Move R)
    5: R_Cong_Err   (Stim R, Block R, Move L)
    6: R_Inc_Corr   (Stim R, Block L, Move R)
    7: R_Inc_Err    (Stim R, Block L, Move L)
    """
    
    # 1. Initialize empty matrices
    n_conds = 8
    models = {
        "Choice (Motor)": np.zeros((n_conds, n_conds)),
        "Prior (Block)":  np.zeros((n_conds, n_conds)),
        "Outcome (Err)":  np.zeros((n_conds, n_conds)),
        "Stimulus (Vis)": np.zeros((n_conds, n_conds))
    }


    idx_move_L = [0, 2, 5, 7] 
    idx_move_R = [1, 3, 4, 6]


    idx_block_L = [0, 1, 6, 7]
    idx_block_R = [2, 3, 4, 5]


    idx_corr = [0, 2, 4, 6]
    idx_err  = [1, 3, 5, 7]

    idx_stim_L = [0, 1, 2, 3]
    idx_stim_R = [4, 5, 6, 7]


    for r in range(n_conds):
        for c in range(n_conds):
            # Choice Model
            if (r in idx_move_L) != (c in idx_move_L):
                models["Choice (Motor)"][r, c] = 1
            
            # Prior Model
            if (r in idx_block_L) != (c in idx_block_L):
                models["Prior (Block)"][r, c] = 1
                
            # Outcome Model
            if (r in idx_corr) != (c in idx_corr):
                models["Outcome (Err)"][r, c] = 1
                
            # Stimulus Model
            if (r in idx_stim_L) != (c in idx_stim_L):
                models["Stimulus (Vis)"][r, c] = 1
    triu_indices = np.triu_indices(n_conds, k=1)
    
    predictors = {}
    for name, matrix in models.items():
        predictors[name] = matrix[triu_indices]
        
    return predictors, list(models.keys())




def run_rsa_regression(accumulated_data, rsa_ideal_matrices=None, normalization=False):

    if rsa_ideal_matrices is None:
        model_vectors, model_names = ideal_rsa_matrices()
    
    X = np.column_stack([model_vectors[name]] for name in model_names]) # type: ignore

    results = {}

    triu_idx = np.triu_indices(8, k=1) # upper triangle indices

    for region in accumulated_data.keys():
        results[region] = {}
        epochs = ['Quiescent', 'Stimulus', 'Choice']

        for epoch in epochs:
            session_matrices = accumulated_data[region][epoch]
            if not session_matrices:
                continue

            pop_matrix = np.vstack(session_matrices)
            
            if normalization:
                pop_matrix = zscore(pop_matrix, axis=1)
                pop_norm = np.nan_to_num(pop_norm)
            
            n_bins = int(pop_matrix.shape[1] / 8)
            reshaped = np.transpose(pop_matrix.reshape(pop_matrix.shape[0], 8, n_bins), (1, 2, 0))
            betas_over_time =  np.zeros((n_bins, len(model_names)))

            reg = LinearRegression(fit_intercept=True)

            for t in range(n_bins):
                trajectories = reshaped[:, t, :]

                y = pdist(trajectories, 'euclidean')

                reg.fit(X, y)
                betas_over_time[t, :] = reg.coef_

        results[region][epoch] = betas_over_time

    return results