import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import pickle as pkl

import ibl_info.rnn_utility_functions as rnn_utility


if __name__ == "__main__":
    iteration = 3000
    session_data = rnn_utility.load_session(
        f"../../ann-rnn-modified/data/adamalltheway/rnn_ann_model_results_10units_{iteration}.pkl"
    )

    trial_side_pid, block_side_pid, action_side_pid, bayes_prior_pid = (
        rnn_utility.information_decomposition_all(session_data)
    )

    data = {}
    data["trial_side"] = trial_side_pid
    data["block_side"] = block_side_pid
    data["action_side"] = action_side_pid
    data["bayes_prior"] = bayes_prior_pid

    with open("../notebooks/{iteration}.pkl", "wb") as f:
        pkl.dump(data, f)
