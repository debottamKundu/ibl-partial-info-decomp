
from one.api import ONE
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.io.one import SpikeSortingLoader, SessionLoader
from brainbox.ephys_plots import plot_brain_regions
from brainbox.behavior.wheel import velocity
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

from ibl_info.broja_pid import compute_pid, coinformation, compute_pid_unbiased, unbiasedMI, MI
from ibl_info.utility import discretize_neural_data, subsample
from ibl_info.prepare_data_pid import compute_intervals, prepare_ephys_data
from ibl_info.load_glm_hmm import load_state_dataframe
import sklearn.linear_model as lm
from sklearn.metrics import make_scorer, accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV



# re-write run single sessions, makes things easier
"""
FLOW:
    1. get session-id
    2. get probes and merge
    3. for each different time interval
        a. compute firing rate for each cluster
        b. throw away clusters below a reasonable threshold
            b1. discard cells with nan or all 0's completely
            b2. see how it looks for a couple of them
            b3. discretize
            b4. plot to see how it looks
        c. then return each cluster based on region
            
        d. compute MI separately
        e. compute PID (set some hacky threshold on MI)
            e1. PID on interval, target variable, units as source.
            e2. PID on interval, one unit as target, others as source.
            e3. For across-regions, treat each region as an aggregate
        f. remember to do error-correction; just curve fitting for now.
        g. once we get the error corrected PIDS,
            for one target variable, we can group them together
                within regions
                across regions.
"""




def write_data(data, session_id, epoch, normalization_params=None,variable='single'):
    if normalization_params==None:
        normalization_params = {'run_correction':'qe', 'normalization':'neuron', 'drop':True, 'mi_calculation':True}

    normalization = '_'.join(str(value) for value in normalization_params.values())
    uuid = session_id + '_'+ epoch + '_'+ normalization + '_' + variable
    
    #  '_'.join(f"{key}: {value}" for key, value in normalization_params.items())
    
    with open(f'D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\interim\{uuid}.pkl','wb') as f:
        pkl.dump(data, f)

#for now keeping all firing rates
def clean_up_neurons(data, params, region_names, percent_of_no_spikes_threshold=.3, firing_rate_threshold=0,plot_flag=True):

    n_regions = len(data)
    cleaned_data = []
    firing_rates = []
    kept_regions = []
    for region in range(n_regions):
        # each portion of the data is in trials x neurons
        # this flips it into neurons x trials
        region_data = data[region].T

        # check for nans
        array_no_nans = region_data[~np.isnan(region_data).any(axis=1)]
        array_no_zeros = array_no_nans[~np.all(array_no_nans == 0, axis=1)]

        # now compute percentage of no spikes
        num_zeros = np.sum(array_no_zeros==0, axis=1)/ array_no_zeros.shape[1]

        # filter such that at least 20 percent of trials have some spiking 
        array_filtered = array_no_zeros[num_zeros <= (1-percent_of_no_spikes_threshold)]
        if len(array_filtered) > 0:
            f_rates = np.mean(array_filtered, axis=1)
            
            f_rate_threshold = np.max(f_rates) * firing_rate_threshold
            keep_idx = f_rates>f_rate_threshold

            array_filtered = array_filtered[keep_idx]
            kept_regions.append(region)
            cleaned_data.append(array_filtered)
            firing_rates.append(f_rates) # keep all of them
            print(f'Kept {len(array_filtered)} out of {len(region_data)}')
            params['region'] = region_names[region]

            if plot_flag==True:
                fig,ax = plt.subplots(figsize=(8,4),ncols=2)
                sns.heatmap(array_filtered,ax =ax[0],cmap='viridis')
                sns.heatmap(region_data, ax=ax[1],cmap='viridis')
                uuid = '_'.join(str(value) for value in params.values())
                fname = f"D:\\personal\\phD\\code\\information-decomposition\\ibl-partial-info-decomp\\reports\\figures\\{uuid}"
                plt.savefig(fname, bbox_inches='tight',facecolor='white')
                plt.suptitle(params['region'])
                plt.close()

    print(f'Kept {len(kept_regions)} out of {n_regions}')
    return cleaned_data, kept_regions, firing_rates
            
def plot_discretized_data(data, region, params):
        
        fig,ax = plt.subplots(figsize=(4,4),ncols=1)

        sns.heatmap(data,ax =ax,cmap='viridis')
        ax.set_title(region)
        uuid = '_'.join(str(value) for value in params.values())
        fname = f"D:\\personal\\phD\\code\\information-decomposition\\ibl-partial-info-decomp\\reports\\figures\\{uuid}_discrete"
        plt.savefig(fname, bbox_inches='tight',facecolor='white')
        plt.close()

def group_regions(data):

    # just compute means
    try:
        grouped_data = np.zeros((len(data), data[0].shape[1]))
    except Exception as e:
        print(len(data))
        print(data)
    for idx in range(len(data)):
        x_data = data[idx]
        mean_of_region = np.ceil(np.mean(x_data, axis=0))
        grouped_data[idx, :] = mean_of_region
    return grouped_data

def generate_source_ids(number_of_neurons):
    combinations_neuronids=[]
    for x in itertools.combinations(range(number_of_neurons), 2):
        combinations_neuronids.append([x[0], x[1]])
    
    combinations_neuronids = np.asarray(combinations_neuronids)
    return combinations_neuronids


def run_decoding(Y, X, hyperparameter_opt=False):
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = lm.LogisticRegression(penalty='l2', max_iter=1000, random_state=42)
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    score = cross_val_score(model, X, Y, cv=kf, scoring='accuracy')
    return np.mean(score)

def compute_running_r2(decoding_variable, neural_data):
    targets = decoding_variable
    model = lm.LogisticRegression(penalty='l2', max_iter=1000, random_state=42)
    global_scores = []

    for i in tqdm(range(1, neural_data.shape[1] + 1)):
        X_subset = neural_data[:, :i]  # Subset of features up to the i-th feature
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = cross_val_score(model, X_subset, targets, cv=kf, scoring=make_scorer(r2_score))
        global_scores.append(np.mean(scores))
    
    # Optionally, print progress for each feature set size
    return global_scores

def decode_epoch(one, session_id, spikes, clusters, epoch, list_of_regions, group=False, normalization_params=None):
    if normalization_params==None:
        normalization_params = {'run_correction':'qe', 'normalization':'neuron', 'drop':True, 'mi_calculation':True}
    
    if epoch=='stim':
        trials, mask = load_trials_and_mask(one, session_id, exclude_nochoice=True, exclude_unbiased=False)
    else:
        trials, mask = load_trials_and_mask(one, session_id, exclude_nochoice=True, exclude_unbiased=True)
    
    trials = trials[mask]

    intervals, decoding_variable = compute_intervals(trials, epoch)
    data_epoch, actual_regions, n_units, cluster_ids = prepare_ephys_data(spikes, clusters, intervals, list_of_regions)
    #trials x neurons
    data = {}
    for region_idx in range(len(actual_regions)):
        print(region_idx, actual_regions[region_idx] )
        if normalization_params['drop']:
            data_discretized = discretize_neural_data(data_epoch[region_idx], normalization_params['normalization'])
        else:
            data_discretized = discretize_neural_data(data_epoch[region_idx].T, normalization_params['normalization']) #TODO: clean this up as well, signature should be the same

        #neurons x trials
        #save plots
        # plot_discretized_data(data_discretized, actual_regions[region_idx], temp_params)
        #now that we have discretized data, we compute the pid
        n_neurons = n_units[region_idx]
        region_name = actual_regions[region_idx]
        print(f'shape: {data_discretized.shape} data')
        global_scores = compute_running_r2(decoding_variable, data_discretized) # 
        data[region_name] = global_scores
    # data is a dict for each region in the mice
    # each entry is equal to the number of neurons in each cluster

    return data



    

def compute_information_decomposition(decoding_variable, neural_data):
    # always same region
    # neural data is in neurons x trials
    targets = decoding_variable
    sources = generate_source_ids(neural_data.shape[0])

    pid_information = np.zeros((len(sources), 4)) # neuronsC2 x 4
    coinformation_data = np.zeros((len(sources), 5)) # neuronsC2 x 4

    for idx in tqdm(range(len(sources)), desc="Running for all sources",leave=False):
        s1 = sources[idx][0]
        s2 = sources[idx][1]
        X1 = np.asarray(neural_data[s1, :], dtype=np.int32)
        X2 = np.asarray(neural_data[s2, :], dtype=np.int32)
        Y = np.asarray(targets, dtype=np.int32)
        u1, u2, red, syn = compute_pid_unbiased(Y, X1, X2)
        coinfo, mi_yx1x2, mi_yx1, mi_yx2 = coinformation(Y, X1, X2)
        X = np.vstack([X1, X2]).T
        decoding_score = run_decoding(Y, X)
        pid_information[idx, :] = u1, u2, red, syn
        coinformation_data[idx,:] = mi_yx1, mi_yx2, coinfo, mi_yx1x2, decoding_score


    # now to organize this?
    # nah, unique information would just be the mean of the first two
    # red and syn  are fine
    # yx1 and yx2 mutual info are also similar to UI
    # the other two are trivariate

    return pid_information, coinformation_data

def compute_mi(decoding_variable, neural_data):
    # neural data is neurons x trials
    # decoding variable is just trials

    mi_data = np.zeros((neural_data.shape[0]))
    for idx in range(len(mi_data)):
        mi_data[idx] = MI(decoding_variable, neural_data[idx, :])
    return mi_data

def decompose_epoch(one, session_id, spikes, clusters, epoch, list_of_regions, group=False, normalization_params=None):
    if normalization_params==None:
        normalization_params = {'run_correction':'qe', 'normalization':'neuron', 'drop':True, 'mi_calculation':True}
    
    plot_flag=  not(group)
     
    if epoch=='stim':
        trials, mask = load_trials_and_mask(one, session_id, exclude_nochoice=True, exclude_unbiased=False)
    else:
        trials, mask = load_trials_and_mask(one, session_id, exclude_nochoice=True, exclude_unbiased=True)
    
    trials = trials[mask]

    intervals, decoding_variable = compute_intervals(trials, epoch)
    data_epoch, actual_regions, n_units, cluster_ids = prepare_ephys_data(spikes, clusters, intervals, list_of_regions)

    # do quality control on data_epoch
    # throw away any cluster that has nans
    # throw away clusters that don't fire at all
    # compute mean firing rate for each region, decide later if we want to threshold or not
    # data epoch is trials x neurons
    if normalization_params['drop'] == True:
        temp_params = {'eid':session_id,'epoch':epoch,'group':group}
        data_epoch, kept_regions, firing_rates = clean_up_neurons(data_epoch, temp_params, actual_regions,plot_flag=plot_flag)
        actual_regions = np.asarray(actual_regions)[kept_regions]
    else:
        firing_rates = None    

    if group and len(actual_regions)>0:        
        data_epoch = group_regions(data_epoch)
        # no Mi corrections
        data_discretized = discretize_neural_data(data_epoch, normalization_params['normalization'])
        pid_info, coinfo = compute_information_decomposition(decoding_variable, data_discretized)
        data = {}
        data['group'] = np.hstack([pid_info, coinfo])
        data['regions'] = actual_regions
        return data

    # discretize 
    data = {}
    for region_idx in range(len(actual_regions)):
        if normalization_params['drop']:
            data_discretized = discretize_neural_data(data_epoch[region_idx], normalization_params['normalization'])
        else:
            data_discretized = discretize_neural_data(data_epoch[region_idx].T, normalization_params['normalization']) #TODO: clean this up as well, signature should be the same

        #neurons x trials
        #save plots
        # plot_discretized_data(data_discretized, actual_regions[region_idx], temp_params)
        #now that we have discretized data, we compute the pid
        n_neurons = n_units[region_idx]
        region_name = actual_regions[region_idx]

        if normalization_params['mi_calculation']==True:
            mi_values = compute_mi(decoding_variable, data_discretized)
            mi_threshold = 0.25 * np.max(mi_values)
            keep_idx = mi_values > mi_threshold
            print(f'MI threshold is {mi_threshold}')
            data_discretized = data_discretized[keep_idx, :]
            pid_info, coinfo = compute_information_decomposition(decoding_variable, data_discretized)
            
        else:
            pid_info, coinfo = compute_information_decomposition(decoding_variable, data_discretized)
        data[region_name] = np.hstack([pid_info, coinfo])
    # data is a dict for each region in the mice
    # each entry is a sources x 9 numpy array    

    return data

def partial_decomposition(one, session_id, list_of_regions, group=False, normalization_params=None):
    if normalization_params==None:
        normalization_params = {'run_correction':'qe', 'normalization':'neuron', 'drop':True, 'mi_calculation':True}
    
    pids, probes = one.eid2pid(session_id)
    if isinstance(probes, list) and len(probes) > 1:
        to_merge = [load_good_units(one, pid=None, eid=session_id, qc=1, pname=probe_name)
                    for probe_name in probes]
        spikes, clusters = merge_probes([spikes for spikes, _ in to_merge], [clusters for _, clusters in to_merge])
    else:   
        spikes, clusters = load_good_units(one, pid=None, eid=session_id, qc=1, pname=probes)
    
    # now we have spikes and good clusters, all together.

    unique_epochs = ['stim']
    #unique_epochs = ['choice']
    for epoch in unique_epochs:
        
        if type(list_of_regions)==dict:
            region = list_of_regions[epoch]
        else:
            region = list_of_regions

        
        #epoch_data = decode_epoch(one, session_id, spikes, clusters, epoch, region, group, normalization_params)
        epoch_data = decompose_epoch(one, session_id, spikes, clusters, epoch, region, group=group, normalization_params=normalization_params)
        if group:
            variable='group_true'
        else:
            variable='group_false'
        write_data(epoch_data, session_id, epoch, normalization_params, variable)


def decompose(one, list_of_regions, list_of_eids):

    for idx in tqdm(range(0,len(list_of_eids)), desc='Running decomposition'):

        session_eid = list_of_eids[idx]
        # entire api signature
        normalization_params = {'run_correction':'qe', 'normalization':'neuron', 'drop':True, 'mi_calculation':False}
        partial_decomposition(one, session_eid, list_of_regions, group=False, normalization_params=normalization_params)

if __name__=='__main__':

    location = 'D:\\personal\\phD\\code\\information-decomposition\\ibl-partial-info-decomp\\data\\processed\\'

    #load eids
    # nice eids covering a span of regions for decoding, glm-hmm
    list_of_eids = np.load('D:\\personal\\phD\\code\\information-decomposition\\ibl-partial-info-decomp\\data\\processed\\eids_with_detailed_insertions_v3.npy',allow_pickle=True)

    # cortical hierarchy eids
    # list_of_eids = np.load(f'{location}minimum_cover_regions_global.npy',allow_pickle=True).item()
    # list_of_eids = np.asarray(list(list_of_eids))

    #load regions
    # change regions here to find nicer ones, maybe have different regions for different decompositions, makes more sense
    # list_of_regions = ['VISpm', 'PRNc', 'IP', 'VISli', 'VM', 'PRNc', 'GRN', 'VM', 'IP',
    #    'APN', 'PPN', 'AUDp', 'PAG', 'PRNc', 'IC', 'SPVI', 'ProS', 'BLA',
    #    'SSp-ll', 'ENTm', 'COAp', 'EPd', 'ProS', 'BLA', 'BMA', 'OT', 'PA',
    #    'MEA', 'EPd', 'BLA']
    
    ### list of top 10 regions for decoding

    # list_of_regions = np.asarray(['VISpm', 'PRNc', 'IP', 'VISli', 'VM', 'SCm', 'GRN', 'LGv', 'VISam',
    #    'PB', 'PRNc', 'GRN', 'VM', 'IP', 'APN', 'PPN', 'VISpm', 'GPe',
    #    'DCO', 'PL', 'PPN', 'AUDp', 'PAG', 'PRNc', 'IC', 'AUDv', 'GRN',
    #    'IP', 'MRN', 'RN'])
    
    # list_of_regions = pd.read_csv(f'{location}global_hierarchy.csv').areas.values
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international")


    # setup done
    list_of_regions = {}
    list_of_regions['stim'] = np.asarray(['VISpm', 'PRNc', 'IP', 'VISli', 'VM', 'SCm', 'GRN', 'LGv', 'VISam',
        'PB'])
    list_of_regions['choice'] = np.asarray(['PRNc', 'GRN', 'VM', 'IP', 'APN', 'PPN', 'VISpm', 'GPe', 'DCO',
        'PL'])
    list_of_regions['feedback'] = np.asarray(['PPN', 'AUDp', 'PAG', 'PRNc', 'IC', 'AUDv', 'GRN', 'IP', 'MRN',
        'RN'])

    decompose(one, list_of_regions, list_of_eids)


    #TODO: do i again compute the redundancy between each regions neuron (lemme think about this)