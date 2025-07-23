from one.api import ONE
from pathlib import Path
import yaml
import os
import wfield
import numpy as np
import pandas as pd
from prior_localization.prepare_data import prepare_widefield
from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import load_trials_and_mask
from prior_localization.functions.utils import compute_mask
from tqdm import tqdm

if __name__ == "__main__":

    one = ONE()
    sessions = one.search(datasets="widefieldU.images.npy")
    print(f"{len(sessions)} sessions with widefield data found")  # type: ignore

    for session_id in tqdm(sessions):  # type: ignore
        sl = SessionLoader(one, eid=session_id)
        sl.load_trials()
        trials_mask = compute_mask(
            sl.trials, align_event="stimOn_times", min_rt=0.08, max_rt=None, n_trials_crop_end=1
        )
        if sum(trials_mask) <= 1:
            raise ValueError(
                f"Session {session_id} has {sum(trials_mask)} good trials, less than 1."
            )
        hemisphere = ("left", "right")
        align_event = "stimOn_times"
        min_rt = 0.08
        max_rt = None
        frame_window = (2, -2)

        data_epoch, actual_regions = prepare_widefield(
            one,
            session_id,
            hemisphere,
            regions="single_regions",
            align_times=sl.trials[align_event].values,
            frame_window=(-3, 3),
            functional_channel=470,
            stage_only=True,
        )

        # NOTE: this will always run, the problem is normally with aligning.
        # Run one on the server to check if we need to fix it or not
