
from one.api import ONE
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from brainbox.ephys_plots import plot_brain_regions
from brainbox.behavior.wheel import velocity
from brainbox.task.trials import get_event_aligned_raster, get_psth
from iblatlas.atlas import AllenAtlas
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from brainbox.behavior.training import compute_performance, plot_psychometric, plot_reaction_time
from brainbox.task.trials import find_trial_ids
from brainbox.io.one import SessionLoader
from pathlib import Path
from brainbox.task.trials import get_event_aligned_raster, get_psth
from brainbox import singlecell
import numpy as np
from iblatlas.atlas import BrainRegions
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import pickle as pkl
from tqdm import tqdm
from pathlib import Path

from ibl_info.broja_pid import compute_pid, coinformation
from ibl_info.utility import discretize_neural_data
from ibl_info.prepare_data_pid import prepare_ephys_data, compute_intervals
from brainwidemap.bwm_loading import merge_probes



def generate_source_ids(number_of_neurons):
    combinations_neuronids=[]
    for x in itertools.combinations(range(number_of_neurons), 2):
        combinations_neuronids.append([x[0], x[1]])
    
    combinations_neuronids = np.asarray(combinations_neuronids)
    return combinations_neuronids


def compute_information_decomposition(decoding_variable, neural_data):
    # always same region
    # neural data is in neurons x trials
    targets = decoding_variable
    sources = generate_source_ids(neural_data.shape[0])

    pid_information = np.zeros((len(sources), 4)) # neuronsC2 x sources
    coinformation_data = np.zeros((len(sources), 4)) # neuronsC2 x sources

    for idx in tqdm(range(len(sources)), desc="Running for all sources",leave=False):
        s1 = sources[idx][0]
        s2 = sources[idx][1]
        X1 = np.asarray(neural_data[s1, :], dtype=np.int32)
        X2 = np.asarray(neural_data[s2, :], dtype=np.int32)
        Y = np.asarray(targets, dtype=np.int32)
        u1, u2, red, syn = compute_pid(Y, X1, X2)
        coinfo, mi_yx1x2, mi_yx1, mi_yx2 = coinformation(Y, X1, X2)
        pid_information[idx, :] = u1, u2, red, syn
        coinformation_data[idx,:] = mi_yx1, mi_yx2, coinfo, mi_yx1x2

    # now to organize this?
    # nah, unique information would just be the mean of the first two
    # red and syn  are fine
    # yx1 and yx2 mutual info are also similar to UI
    # the other two are trivariate

    return pid_information, coinformation_data


def write_data(data, session_id, epoch, normalization):

    uuid = session_id + '_'+ epoch + '_'+ normalization
    with open(f'D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\interim\{uuid}','wb') as f:
        pkl.dump(data, f)
    

def run_decomposition(one, session_id, spikes, clusters, epoch, list_of_regions, normalization='neuron'):

    sl = SessionLoader(one, eid=session_id)
    if epoch=='stim':
        trials, mask = load_trials_and_mask(one, session_id, exclude_nochoice=True, exclude_unbiased=False)
    else:
        trials, mask = load_trials_and_mask(one, session_id, exclude_nochoice=True, exclude_unbiased=True)
    
    trials = trials[mask]

    intervals, decoding_variable = compute_intervals(trials, epoch)
    data_epoch, actual_regions, n_units, cluster_ids = prepare_ephys_data(spikes, clusters, intervals, list_of_regions)

    # now what do we run pid here
    # why not
    #NOTE: For each unit from a region, generate source, targets and compute pid
    #NOTE: normalize the data another way to see variance

    # binned spikes is trials x neurons
    # the discretize function wants neurons x trials
    data = {}
    for region_idx in range(len(actual_regions)):
        data_discretized = discretize_neural_data(data_epoch[region_idx].T, normalization)
        #neurons x trials
        #now that we have discretized data, we compute the pid
        n_neurons = n_units[region_idx]
        region_name = actual_regions[region_idx]

        pid_info, coinfo = compute_information_decomposition(decoding_variable, data_discretized)
        data[region_name] = np.hstack([pid_info, coinfo])

    # data is a dict for each region in the mice
    # each entry is a sources x 8 numpy array    
    return data



def cortical_hierarchy(one, session_id, list_of_regions, normalization='neuron'):

    pids, probes = one.eid2pid(session_id)
    if isinstance(probes, list) and len(probes) > 1:
        to_merge = [load_good_units(one, pid=None, eid=session_id, qc=1, pname=probe_name)
                    for probe_name in probes]
        spikes, clusters = merge_probes([spikes for spikes, _ in to_merge], [clusters for _, clusters in to_merge])
    else:   
        spikes, clusters = load_good_units(one, pid=None, eid=session_id, qc=1, pname=probes)

    # for a particular epoch

    unique_epochs = ['stim','choice','feedback']

    for epoch in tqdm(unique_epochs,leave='False',desc=f'Running for {epoch}'):
        epoch_data = run_decomposition(one, session_id, spikes, clusters, epoch, list_of_regions, normalization)
        write_data(epoch_data, session_id, epoch, normalization)

    # this will generate an pid array for every group of neurons for all the eids we want
    


def calculate_cortical_hierachy(one, list_of_regions, list_of_eids):

    for idx in tqdm(range(len(list_of_eids)),desc=f'Running decomposition on {idx}/{len(list_of_eids)}'):
        
        session_id = list_of_eids[idx]
        cortical_hierarchy(one, session_id, list_of_regions)

        print("-----------------------test done-------------------")
        return



if __name__=='__main__':

    location = 'D:\\personal\\phD\\code\\information-decomposition\\ibl-partial-info-decomp\\data\\processed\\'

    #load eids
    list_of_eids = np.load(f'{location}minimum_cover_regions_global.npy',allow_pickle=True).item()
    list_of_eids = np.asarray(list(list_of_eids))

    #load regions

    list_of_regions = pd.read_csv(f'{location}global_hierarchy.csv').areas.values
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international")


    # setup done
    cortical_hierarchy(one, list_of_regions, list_of_eids)