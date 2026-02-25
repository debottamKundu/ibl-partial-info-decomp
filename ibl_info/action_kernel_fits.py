from joblib import Parallel, delayed
import numpy as np
from behavior_models import models
from one.api import ONE
import brainbox.io.one as bbone
from ibl_info.pseudosession import get_requisite_eids
from ibl_info.utils import check_config
import pickle as pkl

config = check_config()


def process_session(eid, one):
    """Function containing the logic for a single session."""
    try:
        # Load data
        one.load_object(eid, "trials")
        sl = bbone.SessionLoader(one=one, eid=eid)
        sl.load_trials()

        # Instantiate model
        my_model = models.ActionKernel(
            path_to_results="results_behavioral",
            session_uuids=eid,
            df_trials=sl.trials,
            single_zeta=False,
        )

        # Train
        my_model.load_or_train(remove_old=False, adaptive=True)

        # Predict and join
        df_prior = my_model.predict_trials()
        df_trials = sl.trials.join(df_prior, how="left")
        return (eid, df_trials)

    except Exception as e:
        print(f"Error processing {eid}: {e}")
        return (eid, None)


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

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
    )
    global_eid_list = get_requisite_eids(one, important_regions)

    # one.load_object(eid, "trials")
    # sl = bbone.SessionLoader(one=one, eid=eid)
    # sl.load_trials()
    # # instantiate model
    # my_model = models.ActionKernel(
    #     path_to_results="results_behavioral",
    #     session_uuids=eid,
    #     df_trials=sl.trials,
    #     single_zeta=False,
    # )

    # # train - this will save data in the current directory
    # my_model.load_or_train(remove_old=False, adaptive=True)

    # # predict trials and eventually join in the original dataframe
    # df_prior = my_model.predict_trials()
    # df_trials = sl.trials.join(df_prior, how="left")

    results_list = Parallel(n_jobs=-1)(
        delayed(process_session)(eid, one) for eid in global_eid_list
    )

    big_dict = {eid: df for eid, df in results_list if df is not None}  # type: ignore

    with open("./data/processed/all_eids_dict.pkl", "wb") as f:
        pkl.dump(big_dict, f)
