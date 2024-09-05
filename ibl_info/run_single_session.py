
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

from ibl_info.utility import computepid_time_intervals, aggregated_regions_time_intervals

def gather_data_stim(trials_df, spikes_probe0, spikes_probe1, time_window=[0,0.1]):
    correct_trials =  trials_df[trials_df['feedbackType']==1]

    left_stim_trials = correct_trials[correct_trials.contrastLeft>0]
    right_stim_trials = correct_trials[correct_trials.contrastRight>0]
    
    stimon_times = np.concatenate([left_stim_trials.stimOn_times.values, right_stim_trials.stimOn_times.values])
    decoding_variable = np.concatenate([np.ones((left_stim_trials.shape[0])),-1*np.ones((right_stim_trials.shape[0]))])
    events_stim_tw = np.array([stimon_times + time_window[0], stimon_times + time_window[1]]).T

    # Neurons x Trials
    spike_count_stim_probe0, cluster_id_probe0 = get_spike_counts_in_bins(spikes_probe0["times"], spikes_probe0["clusters"], events_stim_tw)
    spike_count_stim_probe1, cluster_id_probe1 = get_spike_counts_in_bins(spikes_probe1["times"], spikes_probe1["clusters"], events_stim_tw)

    # for now, we don't use the single-cell peths

    return spike_count_stim_probe0, spike_count_stim_probe1, cluster_id_probe0, cluster_id_probe1, decoding_variable

def gather_data_choice(trials_df, spikes_probe0, spikes_probe1, time_window=[-0.2,0.0]):
    

    left_choice_trials = trials_df[trials_df.choice==1]
    right_choice_trials = trials_df[trials_df.choice==-1]
    
    choice_times = np.concatenate([left_choice_trials.firstMovement_times.values, right_choice_trials.firstMovement_times.values])
    decoding_variable = np.concatenate([np.ones((left_choice_trials.shape[0])),-1*np.ones((right_choice_trials.shape[0]))])
    events_choice_tw = np.array([choice_times + time_window[0], choice_times + time_window[1]]).T

    # Neurons x Trials
    spike_count_stim_probe0, cluster_id_probe0 = get_spike_counts_in_bins(spikes_probe0["times"], spikes_probe0["clusters"], events_choice_tw)
    spike_count_stim_probe1, cluster_id_probe1 = get_spike_counts_in_bins(spikes_probe1["times"], spikes_probe1["clusters"], events_choice_tw)

    # for now, we don't use the single-cell peths

    return spike_count_stim_probe0, spike_count_stim_probe1, cluster_id_probe0, cluster_id_probe1, decoding_variable

def gather_data_feedback(trials_df, spikes_probe0, spikes_probe1, time_window=[0.0, 0.2]):
    

    correct_feedback_trials = trials_df[trials_df.feedbackType==1]
    incorrect_feedback_trials = trials_df[trials_df.feedbackType==-1]
    
    feedback_times = np.concatenate([correct_feedback_trials.feedback_times.values, incorrect_feedback_trials.feedback_times.values])
    decoding_variable = np.concatenate([np.ones((correct_feedback_trials.shape[0])),-1*np.ones((incorrect_feedback_trials.shape[0]))])
    events_feedback_tw = np.array([feedback_times + time_window[0], feedback_times + time_window[1]]).T

    # Neurons x Trials
    spike_count_stim_probe0, cluster_id_probe0 = get_spike_counts_in_bins(spikes_probe0["times"], spikes_probe0["clusters"], events_feedback_tw)
    spike_count_stim_probe1, cluster_id_probe1 = get_spike_counts_in_bins(spikes_probe1["times"], spikes_probe1["clusters"], events_feedback_tw)

    # for now, we don't use the single-cell peths

    return spike_count_stim_probe0, spike_count_stim_probe1, cluster_id_probe0, cluster_id_probe1, decoding_variable

def gather_data_prior():
    return NotImplementedError

def plot_neurons(neural_data, regions):
    """
    Average Heatmap for neurons locked to a condition

    Args:
        neural_data (np.array): Np array with neurons x trials
    """
    

    fig, ax = plt.subplots(figsize=(8,8))
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    sns.heatmap(neural_data, yticklabels=regions, cmap="Greys", cbar_kws={'shrink':0.5})
    plt.show()


def cleanup_data(neural_data, regions):
    """
    Throw away rows with root, void and other undesirable regions

    Args:
        neural_data (np.array): neurons x trials
        regions (np.array): acronym for neurons
    """

    # if region not in region_info.csv, throw away
    nice_regions = pd.read_csv('../data/external/region_info.csv')['Beryl']
    bad_indices = np.isin(regions, nice_regions)==False
    print(f"Neurons thrown away: {regions[bad_indices]}")

    #TODO: maybe throw away regions that don't have enough neurons

    neural_data = neural_data[~bad_indices, :]
    regions = regions[~bad_indices]

    return np.asarray(neural_data, dtype=np.int32), regions

def generate_sources(number_of_neurons, regions):
    """Generate a combination of all possible pairs

    Args:
        number_of_neurons (int): Number of unique neurons in the dataset
        regions (np.array) : region ids
    """
    combinations = []
    for x in itertools.combinations(range(number_of_neurons), 2):
        combinations.append([regions[x[0]], regions[x[1]]])
    combinations = np.asarray(combinations)
    return combinations

def organise_pid_information(pid_data, neuron_pairs):
    # convert neuron pairs into unique combinations
    # key_array = set()
    # for regions in neuron_pairs:
    #     temp_key = np.sort(regions)
    #     key_array.add(temp_key[0]+'_'+temp_key[1])
    
    # now for each element in the key array, build a dictionary
    data_redundancy = {}
    data_synergy  = {}
    data_unique = {}

    # maybe keep unique information as well

    for idx in range(pid_data.shape[0]): # iterate over all sources
        temp_key = np.sort(neuron_pairs[idx])
        temp_key = temp_key[0]+'_'+temp_key[1]
        key_0 = neuron_pairs[idx][0]
        key_1 = neuron_pairs[idx][1]

        if key_0 in data_unique.keys():
            t = data_unique[key_0]
            t.append(pid_data[idx,0])
            data_unique[key_0] = t
            
        else:
            data_unique[key_0] = [pid_data[idx,0]]
        
        # similarly
        if key_1 in data_unique.keys():
            t = data_unique[key_1]
            t.append(pid_data[idx,1])
            data_unique[key_1] = t
            
        else:
            data_unique[key_1] = [pid_data[idx,1]]
        
        if temp_key in data_redundancy.keys():
            t = data_redundancy[temp_key]
            t.append(pid_data[idx,2])
            data_redundancy[temp_key]= t 
        else:
            data_redundancy[temp_key] = [pid_data[idx,2]]
        
        if temp_key in data_synergy.keys():
            t = data_synergy[temp_key]
            t.append(pid_data[idx,3])
            data_synergy[temp_key] = t
            
        else:
            data_synergy[temp_key] = [pid_data[idx,3]]
        
    return data_unique, data_synergy, data_redundancy

def run_pid_for_sources(decoding_variable, neural_data, regions):
    # now what;
    # compute pid for everything
    targets = decoding_variable
    sources = generate_sources(len(regions), regions)
    # in case i need to create a map
    # region_indices = {string: i for i, string in enumerate(set(regions))}
    # now run pid
    pid_information = np.zeros((len(sources), 4)) # neuronsC2 x sources
    max_length_region_name = np.max(np.vectorize(len)(regions))
    neuron_pairs = np.empty(len(sources),2, dtype=f'U{max_length_region_name}')

    for idx in len(sources):
        s1 = sources[idx][0]
        s2 = sources[idx][1]
        X1 = neural_data[s1, :]
        X2 = neural_data[s2, :]
        Y = targets
        u1, u2, red, syn = computepid_time_intervals(Y, X1, X2)
        pid_information[idx, :] = u1, u2, red, syn
        neuron_pairs[idx, 0] = regions[s1]
        neuron_pairs[idx, 1] = regions[s2]
    
    data_unique, data_synergy, data_redundancy = organise_pid_information(pid_information, neuron_pairs)

    return data_unique, data_synergy, data_redundancy

def combine_probes(spike_count_stim_probe0, spike_count_stim_probe1, regions_probe0, regions_probe1, aggregate=False):
    
    if aggregate:
        # combine neurons from multiple regions into one big chunk
        # spike_count_probe is neurons x trials
        aggregate_neural_data = np.vstack([spike_count_stim_probe0, spike_count_stim_probe1])
        aggregate_regions = np.concatenate([regions_probe0, regions_probe1])

        # now run aggregation
        aggregate_neural_data = aggregated_regions_time_intervals(aggregate_neural_data.T, aggregate_regions) # because the function expects trials x neurons
        # now we clean up the data?
        neural_data, regions = cleanup_data(aggregate_neural_data.T, aggregate_regions)
    else:
        neural_data_probe0, regions_probe0 = cleanup_data(spike_count_stim_probe0, regions_probe0)        
        neural_data_probe1, regions_probe1 = cleanup_data(spike_count_stim_probe1, regions_probe1)
                                                
        # concatenate regions and neural data
        neural_data = np.vstack([neural_data_probe0, neural_data_probe1])
        regions = np.concatenate([regions_probe0, regions_probe1])

    plot_neurons(neural_data, regions)
    return neural_data, regions

def write_to_disk(eid, data_unique, data_synergy, data_redundancy, condition):
    
    with open(f'../data/processed/{eid}_data_unique_{condition}','wb') as f:
        pkl.dump(data_unique, f)
    
    with open(f'../data/processed/{eid}_data_synergy_{condition}','wb') as f:
        pkl.dump(data_synergy, f)

    with open(f'../data/processed/{eid}_data_redundancy_{condition}','wb') as f:
        pkl.dump(data_redundancy, f)

def pid_stim_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False):

    (
        spike_count_stim_probe0, 
        spike_count_stim_probe1, 
        cluster_id_probe0, 
        cluster_id_probe1, 
        decoding_variable
    ) = gather_data_stim(trials_df, spikes_probe0, spikes_probe1)

    regions_probe0 = clusters_probe0['Beryl'][cluster_id_probe0].to_numpy()
    regions_probe1 = clusters_probe1['Beryl'][cluster_id_probe1].to_numpy()

    # drop unwanted neurons
    neural_data, regions = combine_probes(spike_count_stim_probe0, spike_count_stim_probe1, regions_probe0, regions_probe1)    
    data_unique, data_synergy, data_redundancy = run_pid_for_sources(decoding_variable, neural_data, regions)
    write_to_disk(eid, data_unique, data_synergy, data_redundancy, 'stim')

def run_single_eid(one, eid, regions_of_interest=None):
    """
    Runs partial information decompostion from neural data collected from
    different probes for a single eid, filtered by the regions of interest  


    Args:
        one (ONE): Working connection to one
        eid (string): Experiment ID
        regions_of_interest (np.array,, optional): Regions in the mice brain to run the analysis on, Defaults to None, keeping all regions
    """

    pids, probes = one.eid2pid(eid) # get pids
    
    # load data
    spikes_probe0, clusters_probe0 = load_good_units(one, pids[0], probes[0])
    spikes_probe1, clusters_probe1 = load_good_units(one, pids[1], probes[1])

    # don't combine until we send it to the pid calculations
    brainreg = BrainRegions()    
    clusters_probe0['Beryl'] = brainreg.acronym2acronym(clusters_probe0.acronym.values, mapping='Beryl')
    clusters_probe1['Beryl'] = brainreg.acronym2acronym(clusters_probe1.acronym.values, mapping='Beryl')


    # load trials and mask
    session_l = SessionLoader(one=one, eid=eid)  
    session_l.load_trials()

    trials_df, trials_mask = load_trials_and_mask(one=one, eid=eid, sess_loader=session_l, min_rt=0.08, max_rt=2.0, nan_exclude='default', exclude_nochoice=True)
    # subset trials based on trials_mask
    trials_df = trials_df[trials_mask]

    # run partial information decompositions
    pid_stim_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1)





    