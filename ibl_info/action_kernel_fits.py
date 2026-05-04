from joblib import Parallel, delayed
import numpy as np
from behavior_models import models
from one.api import ONE
import brainbox.io.one as bbone
from ibl_info.pseudosession import get_requisite_eids
from ibl_info.utils import check_config
import pickle as pkl
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units

config = check_config()


def process_session(session_id):

    try:

        one = ONE(
            mode="local",
        )
        print(session_id)
        trials, mask = load_trials_and_mask(
            one,
            session_id,
            exclude_nochoice=True,  # True
            exclude_unbiased=False,  # should include no-choice trials
            min_rt=0.02,
        )

        my_model = models.ActionKernel(
            path_to_results="results_behavioral_zeta",
            session_uuids=session_id,
            df_trials=trials,
            single_zeta=True,  # should be True?
        )

        my_model.load_or_train(remove_old=False, adaptive=True)

        df_prior = my_model.predict_trials()
        df_trials = trials.join(df_prior, how="left")
        return (session_id, df_trials)

    except Exception as e:
        print(f"Error processing {session_id}: {e}")
        return (session_id, None)


if __name__ == "__main__":

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
    )
    # global_eid_list = get_requisite_eids(one, important_regions)

    bwm_df = bwm_query(one)
    global_eid_list = bwm_df["eid"].unique()

    workers = 32

    results_list = Parallel(n_jobs=-1)(
        delayed(process_session)(eid, one) for eid in global_eid_list
    )

    big_dict = {eid: df for eid, df in results_list if df is not None}  # type: ignore

    with open("./data/processed/all_eids_dict_single_zeta_complete_bwm.pkl", "wb") as f:
        pkl.dump(big_dict, f)
