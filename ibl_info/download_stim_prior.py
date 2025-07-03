from ibl_info.utility import download_data
from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap import bwm_units, bwm_query
import pandas as pd
from tqdm import tqdm


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

    for eid in tqdm(eids):

        try:
            pids, probes = one.eid2pid(eid)

            for pid in pids:

                download_data(one, pid)
        except Exception as e:
            print(e)
            print(f"Downloading failed for spike sorting data, eid {eid}, skipping")
