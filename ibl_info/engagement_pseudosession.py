# loads up widefield and ephys eids
# generate 100 pseudosessions for each eid
# check if action-kernel fit exists, if so generate choices, otherwise fit and generate choices
# save dict

from joblib import Parallel, delayed
import numpy as np
from behavior_models import models
from one.api import ONE
import brainbox.io.one as bbone
from ibl_info.pseudosession import get_requisite_eids
from ibl_info.utils import check_config
import pickle as pkl
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from behavior_models import models
from behavior_models.utils import format_input as mut_format_input
from brainbox.task.closed_loop import generate_pseudo_session
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

config = check_config()


def pseudosession_per_eid(session_id, subject_name, n_permutations=100):

    one = ONE(
        # base_url="https://openalyx.internationalbrainlab.org",
        # password="international",
        # silent=True,
        # username="intbrainlab",
        mode="local",
    )
    print(session_id)
    print(subject_name)
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

    arr_params = my_model.get_parameters(parameter_type="posterior_mean")[None]
    pseudosession_array = []
    for idx in range(n_permutations):
        pseudosess = generate_pseudo_session(trials, generate_choices=False)
        stim, _, side = mut_format_input(  # pyright: ignore[reportAssignmentType]
            [pseudosess.signed_contrast.values],
            [trials.choice.values],
            [pseudosess.stim_side.values],
        )
        act_sim, stim, side = my_model.simulate(  # type: ignore
            arr_params, stim[0, :], side[0, :], nb_simul=1, only_perf=False, return_prior=False
        )
        act_sim = np.array(act_sim.squeeze().T, dtype=np.int64)
        sessionx = pseudosess.assign(choice=act_sim)
        sessionx = sessionx.assign(subject=subject_name)
        sessionx = sessionx.assign(session_number=idx)
        sessionx = sessionx.assign(session_id=session_id)
        pseudosession_array.append(sessionx)

    return pd.concat(pseudosession_array)  # list of pandas dataframes.


def prepare_and_run(args):

    session_id, subject_name = args
    savepath = f"./data/generated/engagement_pseudosessions/{session_id}.pqt"
    if os.path.exists(savepath):
        return savepath

    try:
        data = pseudosession_per_eid(session_id, subject_name)
        df = pd.DataFrame(data)
        df.to_parquet(savepath)
        return savepath
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":

    important_regions = config["stim_prior_regions"]
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
    )
    # global_eid_list = get_requisite_eids(one, important_regions)

    # get wifi sessions
    # wifi_sessions = one.search(datasets="widefieldU.images.npy")
    # print(f"{len(wifi_sessions)} sessions with widefield data found")  # type: ignore

    # temp_array = []
    # for eid in wifi_sessions:  # type: ignore
    #     temp_array.append(str(eid))
    # wifi_sessions = np.asarray(temp_array)

    # global_eid_list = np.concatenate([global_eid_list, wifi_sessions])  # type: ignore
    # global_eid_list = np.unique(global_eid_list)

    bwm_df = bwm_query(one)
    global_eid_list = bwm_df["eid"].unique()

    # leftover_eids = list(set(bwm_df["eid"].unique()).difference(set(global_eid_list)))
    # create global subject list
    workers = 32  # (os.cpu_count()) // 2  # type: ignore

    # now we need to generate pseudosessions and fit
    subject_list = []
    for eid in global_eid_list:  # NOTE: check before running
        details = one.get_details(eid)
        subject = details["subject"]  # type: ignore
        subject_list.append(subject)

    all_tasks_to_run = list(
        zip(global_eid_list, subject_list)
    )  # NOTE: check eid list provided here before running

    # run a single one
    # prepare_and_run(all_tasks_to_run[0])

    run_all_flag = True
    if run_all_flag:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(prepare_and_run, t) for t in all_tasks_to_run]

            # We just collect the paths now, not the data frames
            saved_paths = []
            for f in tqdm(as_completed(futures), total=len(futures)):
                res = f.result()
                if res:
                    saved_paths.append(res)
    print("Fin.")
