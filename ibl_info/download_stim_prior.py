from ibl_info.utils import download_data
from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap import bwm_units, bwm_query
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp


def process_eid(eid):
    """
    This function contains the core logic for each eid.
    It will be executed by a worker process.
    """
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international")
    try:
        pids, probes = one.eid2pid(eid)
        for pid in pids:
            download_data(one, pid)
        return f"Downloading successful for eid {eid}"
    except Exception as e:
        return f"Downloading failed for spike sorting data, eid {eid}, skipping. Error: {e}"


if __name__ == "__main__":

    # imported from notebook joint_stimulus_prior_regions.ipynb
    important_regions = [
        "MOs",
        "SSp-ul",
        "VISp",
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

    one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international")

    bwm_df = bwm_query(one)
    unit_df = bwm_units(one)

    units_regions_of_interest = unit_df[unit_df["Beryl"].isin(important_regions)]
    eids = units_regions_of_interest["eid"].unique()

    # find the differences
    leftovereids = list((set(bwm_df["eid"].unique())).difference(set(eids)))

    num_processes = mp.cpu_count() // 2

    with mp.Pool(processes=num_processes) as pool:
        # Use pool.imap_unordered for a tqdm progress bar.
        # It yields results as they are completed, so the progress bar
        # updates as soon as a process finishes, regardless of order.
        results = list(
            tqdm(pool.imap_unordered(process_eid, leftovereids), total=len(leftovereids))
        )

    # Print the results from each process
    for result in results:
        print(result)
