"""
    Randomly select mice for train and test.
"""
import pandas as pd
import numpy as np
import json

alt_set = True  # create an alternative split

np.random.seed(9)  # small search (from 5 to 9) over seeds which split large unique session types relatively evenly, see data_playground.py
print("This code is only supposed to run once, you sure to re-run?")
response = input()
if response != "I really want to run again":
    quit()

file = "./processed_data/all_mice.csv"
mice_data = pd.read_csv(file, low_memory=False)

# train and test split
sessions, trials_per_session = np.unique(mice_data.session, return_counts=True)
n_sessions = len(sessions)

# sanity check:
assert (mice_data.groupby('session')['response_times'].nunique().values == trials_per_session).all()

criteria = np.zeros((4, len(sessions)))
for i, session in enumerate(sessions):
    data = mice_data[mice_data.session == session]
    criteria[0, i] = len(data)
    if len(data) > 400:
        criteria[1, i] = (data.stimOn_times.values[399] - data.stimOn_times.values[0]) / 60
    else:
        criteria[1, i] = 90  # if session didn't even reach 400 trials, just say it took 90 minutes
        # print((data.stimOn_times.values[-1] - data.stimOn_times.values[0]) / 60, np.mean((data.feedbackType + 1) / 2), len(data))
    criteria[2, i] = np.mean((data.feedbackType + 1) / 2)
    criteria[3, i] = ((data[np.logical_or(data.contrastLeft == 1., data.contrastRight == 1.)].feedbackType + 1) / 2).mean()

import matplotlib.pyplot as plt

def hists(mask, title):
    unmask = ~mask
    fs = 18
    plt.figure(figsize=(12, 6.75))
    plt.subplot(311)
    plt.title(title, fontsize=fs)
    plt.hist([criteria[0, mask], criteria[0, unmask]], color=['r', 'k'], stacked=True, bins=np.arange(50, 1550, 50))
    plt.annotate("n={}".format(mask.sum()), (0.1, 0.85), xycoords='figure fraction', color='r', fontsize=fs)
    plt.annotate("n={}".format(unmask.sum()), (0.85, 0.85), xycoords='figure fraction', color='k', fontsize=fs)
    plt.xlabel("# of trials", fontsize=fs)

    plt.subplot(312)
    plt.hist([criteria[2, mask], criteria[2, unmask]], color=['r', 'k'], stacked=True, bins=np.linspace(0.34, 1, 34))
    plt.ylabel("# of sessions", fontsize=fs)
    plt.xlabel("% correct", fontsize=fs)

    plt.subplot(313)
    plt.hist([criteria[3, mask], criteria[3, unmask]], color=['r', 'k'], stacked=True, bins=np.linspace(0.34, 1, 34))
    plt.xlabel("% correct on easy", fontsize=fs)

    plt.tight_layout()
    plt.savefig(title)
    plt.close()

mask = criteria[0] < 400
hists(mask, "Criterion: Less than 400 trials")

mask = criteria[1] > 45
hists(mask, "Criterion: 400 trials after 45 minutes")

mask = criteria[3] < 0.9
hists(mask, "Criterion: Performance on easy < 90%")

mask = np.logical_or(criteria[3] < 0.9, criteria[0] < 400)
hists(mask, "Criterion: Easy performance bad or too few trials")

mask = np.logical_or(criteria[3] < 0.9, criteria[1] > 45)
hists(mask, "Criterion: Easy performance bad or too slow")

# Use BWM criterion
mask = ~np.logical_or(criteria[3] < 0.9, criteria[0] < 400)
sessions, trials_per_session = sessions[mask], trials_per_session[mask]

assert sessions.shape[0] == 539, "Something changed"

print("Less than 400 trials: {}".format((trials_per_session < 400).sum()))
print("Exactly 400 trials: {}".format((trials_per_session == 400).sum()))
print("Exactly 401 trials: {}".format((trials_per_session == 401).sum()))
print("Exactly 402 trials: {}".format((trials_per_session == 402).sum()))
print("Exactly 403 trials: {}".format((trials_per_session == 403).sum()))
print("Exactly 404 trials: {}".format((trials_per_session == 404).sum()))
print("Exactly 405 trials: {}".format((trials_per_session == 405).sum()))
print("Exactly 406 trials: {}".format((trials_per_session == 406).sum()))
print("Exactly 407 trials: {}".format((trials_per_session == 407).sum()))
print("More than 407 trials: {}".format((trials_per_session > 407).sum()))
# Less than 400 trials: 0
# Exactly 400 trials: 1
# Exactly 401 trials: 14
# Exactly 402 trials: 3
# Exactly 403 trials: 5
# Exactly 404 trials: 3
# Exactly 405 trials: 2
# Exactly 406 trials: 1
# Exactly 407 trials: 2
# More than 407 trials: 508


# we treat everything below 406 as a separate bin, to catch any right after 400 trial weirdness
bound1, bound2, bound3 = np.quantile(trials_per_session[trials_per_session >= 405], q=[0.25, 0.5, 0.75])

train_eids, test_eids, validate_eids = [], [], []

short_sessions = sessions[trials_per_session <= 405]
np.random.shuffle(short_sessions)
print(short_sessions.size / 539, short_sessions.shape)
if not alt_set:
    test_eids += list(short_sessions[:4])
    validate_eids += list(short_sessions[4:8])
    train_eids += list(short_sessions[8:])
else:
    train_eids += list(short_sessions[:4]) + list(short_sessions[8:-4])
    validate_eids += list(short_sessions[4:8])
    test_eids += list(short_sessions[-4:])

quart1 = sessions[np.logical_and(405 < trials_per_session, trials_per_session <= bound1)]
np.random.shuffle(quart1)
print(quart1.size / 539, quart1.shape)
if not alt_set:
    test_eids += list(quart1[:16])
    validate_eids += list(quart1[16:32])
    train_eids += list(quart1[32:])
else:
    train_eids += list(quart1[:16]) + list(quart1[32:-16])
    validate_eids += list(quart1[16:32])
    test_eids += list(quart1[-16:])

quart2 = sessions[np.logical_and(bound1 < trials_per_session, trials_per_session <= bound2)]
np.random.shuffle(quart2)
print(quart2.size / 539, quart2.shape)
if not alt_set:
    test_eids += list(quart2[:16])
    validate_eids += list(quart2[16:32])
    train_eids += list(quart2[32:])
else:
    train_eids += list(quart2[:16]) + list(quart2[32:-16])
    validate_eids += list(quart2[16:32])
    test_eids += list(quart2[-16:])

quart3 = sessions[np.logical_and(bound2 < trials_per_session, trials_per_session <= bound3)]
np.random.shuffle(quart3)
print(quart3.size / 539, quart3.shape)
if not alt_set:
    test_eids += list(quart3[:16])
    validate_eids += list(quart3[16:32])
    train_eids += list(quart3[32:])
else:
    train_eids += list(quart3[:16]) + list(quart3[32:-16])
    validate_eids += list(quart3[16:32])
    test_eids += list(quart3[-16:])

quart4 = sessions[bound3 < trials_per_session]
np.random.shuffle(quart4)
print(quart4.size / 539, quart4.shape)
if not alt_set:
    test_eids += list(quart4[:16])
    validate_eids += list(quart4[16:32])
    train_eids += list(quart4[32:])
else:
    train_eids += list(quart4[:16]) + list(quart4[32:-16])
    validate_eids += list(quart4[16:32])
    test_eids += list(quart4[-16:])

assert len(train_eids) == 403 and len(test_eids) == 68 and len(validate_eids) == 68, "Numbers got messed up"  # now we're doing a 72.5, 13.75, 13.75 split, to take 4 from each quartile

# check for mutual exclusivity
for n in train_eids:
    assert n not in test_eids
    assert n not in validate_eids

for n in test_eids:
    assert n not in train_eids
    assert n not in validate_eids

for n in validate_eids:
    assert n not in test_eids
    assert n not in train_eids

assert np.unique(train_eids + test_eids + validate_eids).size == 539, "Number of unique eids is not 539"

# save split names
if not alt_set:
    json.dump(list(train_eids), open("train_eids", 'w'))
    json.dump(list(test_eids), open("test_eids", 'w'))
    json.dump(list(validate_eids), open("validate_eids", 'w'))
else:
    json.dump(list(train_eids), open("train_eids_alt", 'w'))
    json.dump(list(test_eids), open("test_eids_alt", 'w'))