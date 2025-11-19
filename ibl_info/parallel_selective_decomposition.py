# the main difference is that it flattens all the tasks
# runs them parallely, waits for the processes to finish, and then collates it
# what is a probable bottleneck: not really bottleneck, but call units_df repeatedly (i think)


from one.api import ONE
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from brainbox.ephys_plots import plot_brain_regions
from iblatlas.atlas import AllenAtlas
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from brainwidemap.bwm_loading import merge_probes
from brainbox.behavior.training import compute_performance, plot_psychometric, plot_reaction_time
from brainbox.task.trials import find_trial_ids
from brainbox.io.one import SessionLoader
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
from sklearn.ensemble import RandomForestClassifier
import os
import concurrent.futures
import functools
from ibl_info.selective_decomposition import run_analysis_single_session, filter_eids
from ibl_info.utils import check_config

config = check_config()


def prepare_and_run_data(task_tuple):

    eid, region, epoch, discretizer = task_tuple
    single_cell_filter = config["single_cell_filter"]
    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
    )
    try:
        # ideally information pickle, but i want to subsample mutliple times
        information_pickle = run_analysis_single_session(
            eid,
            epoch,
            one,
            region,
            discretize_method=discretizer,
            single_cell_filter=single_cell_filter,
        )
        if information_pickle == {}:
            return region, eid, None
        else:
            return region, eid, information_pickle
    except Exception as e:
        print(f"Error regarding {eid} in region {region}: {e}")
        return region, eid, None


def run_flattened(list_of_regions, epoch, discretizer):

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        username="intbrainlab",
        password="international",
    )
    unit_df = bwm_units(one)
    all_tasks_to_run = []
    for region in list_of_regions:
        selective_eids = filter_eids(unit_df, region, significant_filter=config["decoder_filter"])
        for eid in tqdm(selective_eids):
            all_tasks_to_run.append((eid, region, epoch, discretizer))

    print(f"Total tasks: {len(all_tasks_to_run)}")

    processed_results = []
    workers = os.cpu_count() // 4  # type: ignore
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results_iterator = executor.map(prepare_and_run_data, all_tasks_to_run)

        processed_results = list(
            tqdm(results_iterator, total=len(all_tasks_to_run), desc="Processing Tasks")
        )

    print("Now collecting and writing out")

    region_data = {}
    for region, eid, information_pickle in processed_results:

        if region not in region_data:
            region_data[region] = {}

        if information_pickle is not None:
            region_data[region][eid] = information_pickle

    if config["single_cell_filter"]:
        suffix = "filtered"
    else:
        suffix = "unfiltered"

    if config["decoder_filter"]:
        suffix += "_significant"
    else:
        suffix += "_all"

    if discretizer == 1:
        suffix += "_alternate"
    else:
        suffix += "_equipopulated"

    n_bins = config["n_bins"]
    suffix += f"_{n_bins}"

    # this will make one huge pickle:
    for region, region_pickle in region_data.items():
        with open(
            f"./data/generated/selective_decomposition_{region}_{epoch}_{suffix}.pkl", "wb"
        ) as f:
            pkl.dump(region_pickle, f)

    print("Done!")


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
    
    discretizer = config["discretize"]
    run_flattened(important_regions, "choice", discretizer=discretizer)
    # 1 is the alternate method
    # can i somehow use the nbins?

    # three random regions; one that has only stim but no prior; one prior but no stim, one just choice
    # random_regions = ["SCs", "VISa", "PO"]
    # run_flattened(random_regions, "stim")
