from brainbox.task.closed_loop import generate_pseudo_session
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from one.api import ONE
from pathlib import Path
from brainbox.task.trials import get_event_aligned_raster, get_psth
from brainbox.singlecell import bin_spikes2D
import numpy as np
from iblatlas.atlas import BrainRegions
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
import warnings
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from behavior_models import models
from behavior_models.utils import format_input as mut_format_input
import torch
from ibl_info.selective_decomposition import filter_eids
from ibl_info.utils import check_config

# just train all action kernel models on the server
config = check_config()


def get_requisite_eids(important_regions):

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
    )
    unit_df = bwm_units(one)

    global_eid_list = []
    for region in important_regions:
        selective_eids = filter_eids(unit_df, region)
        global_eid_list.extend(selective_eids)

    global_eid_list = np.asarray(global_eid_list)
    global_eid_list = np.unique(global_eid_list)

    return global_eid_list


def generate_pseudosession(one, eid, choice=False):
    trials, mask = load_trials_and_mask(
        one, eid, exclude_nochoice=True, exclude_unbiased=True
    )  # to keep same statistics
    trials = trials[mask]
    pseudosess = generate_pseudo_session(trials, generate_choices=False)
    if choice == False:
        return pseudosess
    stim, _, side = mut_format_input(  # pyright: ignore[reportAssignmentType]
        [pseudosess.signed_contrast.values],
        [trials.choice.values],
        [pseudosess.stim_side.values],
    )

    model = models.ActionKernel(
        path_to_results=config["model_locations"],
        mouse_name=eid,
        session_uuids=eid,
        df_trials=trials,
        single_zeta=True,
    )
    model.load_or_train(remove_old=False, adaptive=True)

    arr_params = model.get_parameters(parameter_type="posterior_mean")[None]

    act_sim, stim, side = model.simulate(  # type: ignore
        arr_params, stim[0, :], side[0, :], nb_simul=1, only_perf=False, return_prior=False
    )
    act_sim = np.array(act_sim.squeeze().T, dtype=np.int64)
    pseudosess["choice"] = act_sim

    return pseudosess


def fit_eid(one, eid):

    trials, mask = load_trials_and_mask(one, eid, exclude_nochoice=False, exclude_unbiased=False)
    trials = trials[mask]
    metadata = {"subject": eid, "eid": eid, "probe_name": "probe00"}

    model = models.ActionKernel(
        path_to_results=config["model_locations"],
        mouse_name=eid,
        session_uuids=eid,
        df_trials=trials,
        single_zeta=True,
    )
    model.load_or_train(remove_old=False, adaptive=True)


if __name__ == "__main__":

    # this flow is not the best, but we can also crosscheck if the behavioral fit exists, so in that way is optimum
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

    subject_id = "CSH_ZAD_022"
    eid = "a82800ce-f4e3-4464-9b80-4c3d6fade333"
    session_id = eid
    one = ONE()
    trials, mask = load_trials_and_mask(
        one, session_id, exclude_nochoice=True, exclude_unbiased=True
    )
    trials = trials[mask]

    model = models.ActionKernel(
        path_to_results=config["model_locations"],
        mouse_name=eid,
        session_uuids=eid,
        df_trials=trials,
        single_zeta=True,
    )
    model.load_or_train(remove_old=False, adaptive=True)
