import os
import pickle as pkl
from tqdm import tqdm
from one.api import ONE
from brainbox.io.one import SessionLoader
from brainbox.behavior.training import compute_performance
import numpy as np
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from brainbox.ephys_plots import plot_brain_regions
from brainbox.task.trials import get_event_aligned_raster, get_psth
from iblatlas.atlas import AllenAtlas
from brainbox.behavior.training import compute_performance, plot_psychometric, plot_reaction_time
import numpy as np

# setup internal ibl connections
one = ONE(base_url="https://alyx.internationalbrainlab.org/")
query = "field_of_view__imaging_type__name,mesoscope"
trained_animals_eids, trained_animals_details = one.search(task_protocol="biasedChoiceWorld", procedures="Imaging", details=True, django=query)  # type: ignore
in_training_animals_eids, in_training_details = one.search(task_protocol="trainingChoiceWorld", procedures="Imaging", details=True, django=query)  # type: ignore

training_subjects = []
for idx in range(len(in_training_details)):  # type: ignore
    training_subjects.append(in_training_details[idx]["subject"])  # type: ignore


trained_subjects = []
for idx in range(len(trained_animals_details)):  # type: ignore
    trained_subjects.append(trained_animals_details[idx]["subject"])  # type: ignore

training_subjects, training_session_counts = np.unique(training_subjects, return_counts=True)
trained_subjects, trained_session_counts = np.unique(trained_subjects, return_counts=True)
animalsofinterest = np.intersect1d(trained_subjects, training_subjects)


for subject in tqdm(animalsofinterest, desc="subject"):

    if os.path.exists(f"../data/processed/{subject}.pkl"):
        continue

    global_perf = []
    global_contrasts = []
    global_probability = []
    sess_details = []
    eids, details = one.search(subject=subject, procedures="Imaging", details=True, django=query)  # type: ignore
    for idx in tqdm(reversed(range(len(eids)))):  # type: ignore
        try:
            sess = SessionLoader(one, eid=eids[idx])  # type: ignore
            sess.load_trials()  # set some collection values
            performance, contrasts, n_contrasts = compute_performance(sess.trials)
            global_perf.append(performance)
            global_contrasts.append(contrasts)
            global_probability.append(np.unique(sess.trials.probabilityLeft))
            sess_details.append(details[idx]["task_protocol"])  # type: ignore
        except Exception as e:
            print(e)
    temp_pickle = {
        "performance": global_perf,
        "contrasts": global_contrasts,
        "probabilities": global_probability,
        "protocol": sess_details,
    }
    with open(f"../data/processed/{subject}.pkl", "wb") as f:
        pkl.dump(temp_pickle, f)
