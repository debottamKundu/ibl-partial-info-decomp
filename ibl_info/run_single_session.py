
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
from ibl_info.utility import discretize_neural_data, subsample
from ibl_info.prepare_data_pid import gather_data_choice, gather_data_feedback, gather_data_stim, combine_probes, gather_data_prior
from ibl_info.load_glm_hmm import load_state_dataframe


def plot_neurons(neural_data, regions, eid, condition, aggregate):
    """
    Average Heatmap for neurons locked to a condition

    Args:
        neural_data (np.array): Np array with neurons x trials
    """
    if aggregate==False:
        agr = ''
    else:
        agr = 'agr'

    fig, ax = plt.subplots(figsize=(8,8))
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    sns.heatmap(neural_data, yticklabels=regions, cmap="Greys", cbar_kws={'shrink':0.5})
    plt.savefig(f'D:/personal/phD/code/information-decomposition/ibl-partial-info-decomp/reports/figures/{eid}_{condition}_{agr}.png')
    plt.close()
    # plt.show()


def generate_sources(number_of_neurons, regions):
    """Generate a combination of all possible pairs

    Args:
        number_of_neurons (int): Number of unique neurons in the dataset
        regions (np.array) : region ids
    
    Returns:
        combination_regions (np.array) : All possible pairs of neurons
        combination_neuronids (np.array) : Pairs of neurons wth ids specified
    """
    combinations_regions = []
    combinations_neuronids = []
    for x in itertools.combinations(range(number_of_neurons), 2):
        combinations_regions.append([regions[x[0]], regions[x[1]]])
        combinations_neuronids.append([x[0], x[1]])
    combinations_regions = np.asarray(combinations_regions)
    combinations_neuronids = np.asarray(combinations_neuronids)
    return combinations_regions, combinations_neuronids

def organise_pid_information(pid_data, neuron_pairs):
    """
    Combine PID information into unique, synergistic and redundant information for all neuron pairs

    Args:
        pid_data (np.array): PID decomposition for all possible pair of sources
        neuron_pairs (np.array): Names of regions

    Returns:
       data_unique (dict): Unique information for each neuron
       data_redundant (dict): Shared inforation between each neuron pair, ordered by region
       data_synergy (dict): Synergistic inforation between each neuron pair, ordered by region
    """
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

def organize_coninfo(coinformation_data, neuron_pairs):
    """
    Combine coinformation and mutual information into usable data for neuron pairs

    Args:
        pid_data (np.array): PID decomposition for all possible pair of sources
        neuron_pairs (np.array): Names of regions

    Returns:
       data_neuron (dict): Mutual information for each neuron
       data_coninfo (dict): Coninformation between each neuron pair, ordered by region
       data_trimI (dict): Mutual information between each neuron pair, ordered by region
    """

    data_neuron = dict()
    data_coninfo = dict()
    data_trimI = dict()

    for idx in range(coinformation_data.shape[0]): # iterate over all sources
        temp_key = np.sort(neuron_pairs[idx])
        temp_key = temp_key[0]+'_'+temp_key[1]
        key_0 = neuron_pairs[idx][0]
        key_1 = neuron_pairs[idx][1]

        if key_0 in data_neuron.keys():
            t = data_neuron[key_0]
            t.append(coinformation_data[idx,0])
            data_neuron[key_0] = t
            
        else:
            data_neuron[key_0] = [coinformation_data[idx,0]]
        
        # similarly
        if key_1 in data_neuron.keys():
            t = data_neuron[key_1]
            t.append(coinformation_data[idx,1])
            data_neuron[key_1] = t
            
        else:
            data_neuron[key_1] = [coinformation_data[idx,1]]
        
        if temp_key in data_coninfo.keys():
            t = data_coninfo[temp_key]
            t.append(coinformation_data[idx,2])
            data_coninfo[temp_key]= t 
        else:
            data_coninfo[temp_key] = [coinformation_data[idx,2]]
        
        if temp_key in data_trimI.keys():
            t = data_trimI[temp_key]
            t.append(coinformation_data[idx,3])
            data_trimI[temp_key] = t
            
        else:
            data_trimI[temp_key] = [coinformation_data[idx,3]]

    return data_neuron, data_coninfo, data_trimI

def run_pid_for_sources(decoding_variable, neural_data, regions):
    """
    Run partial information decomposition for all sources and a single target

    Args:
        decoding_variable (np.array): Target variable, choice, stim, feedback or prior
        neural_data (np.array): neurons x trials
        regions (np.array): names of regions for each neuron

    Returns:
       data_unique (dict): Unique information for each neuron
       data_redundant (dict): Shared inforation between each neuron pair, ordered by region
       data_synergy (dict): Synergistic inforation between each neuron pair, ordered by region
    """
    # now what;
    # compute pid for everything
    targets = decoding_variable
    neuron_pairs, sources = generate_sources(len(regions), regions)
    # in case i need to create a map
    # region_indices = {string: i for i, string in enumerate(set(regions))}
    # now run pid
    pid_information = np.zeros((len(sources), 4)) # neuronsC2 x sources
    coinformation_data = np.zeros((len(sources), 4)) # neuronsC2 x sources
    #max_length_region_name = np.max(np.vectorize(len)(regions))
    #neuron_pairs = np.empty((len(sources),2), dtype=f'U{max_length_region_name}')

    for idx in tqdm(range(len(sources)), desc="Running for all sources"):
        s1 = sources[idx][0]
        s2 = sources[idx][1]
        X1 = np.asarray(neural_data[s1, :], dtype=np.int32)
        X2 = np.asarray(neural_data[s2, :], dtype=np.int32)
        Y = np.asarray(targets, dtype=np.int32)
        u1, u2, red, syn = compute_pid(Y, X1, X2)
        coinfo, mi_yx1x2, mi_yx1, mi_yx2 = coinformation(Y, X1, X2)
        pid_information[idx, :] = u1, u2, red, syn
        coinformation_data[idx,:] = mi_yx1, mi_yx2, coinfo, mi_yx1x2
        # coninfo and mi_yx1x2 are for triplets, the other two are for single neurons
        
        # neuron_pairs[idx, 0] = regions[s1]
        # neuron_pairs[idx, 1] = regions[s2]
    
    data_unique, data_synergy, data_redundancy = organise_pid_information(pid_information, neuron_pairs)
    data_neuron, data_coninfo, data_trimI = organize_coninfo(coinformation_data, neuron_pairs)

    return data_unique, data_synergy, data_redundancy, data_neuron, data_coninfo, data_trimI




def write_to_disk(eid, data_unique, data_synergy, data_redundancy, data_neuron, data_coninfo, data_trimI, condition, aggregate, average=''):
    
    if aggregate==False:
        agr = ''
    else:
        agr = 'agr'

    with open(f'D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\processed\{eid}_data_unique_{condition}{agr}_{average}','wb') as f:
        pkl.dump(data_unique, f)
    
    with open(f'D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\processed\{eid}_data_synergy_{condition}{agr}_{average}','wb') as f:
        pkl.dump(data_synergy, f)

    with open(f'D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\processed\{eid}_data_redundancy_{condition}{agr}_{average}','wb') as f:
        pkl.dump(data_redundancy, f)

    # with open(f'D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\processed\{eid}_mutual_information_{condition}{agr}_{average}','wb') as f:
    #     pkl.dump(mutual_information, f)

    with open(f'D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\processed\{eid}_data_neuron_{condition}{agr}_{average}','wb') as f:
        pkl.dump(data_neuron, f)

    with open(f'D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\processed\{eid}_data_coninfo_{condition}{agr}_{average}','wb') as f:
        pkl.dump(data_coninfo, f)
    
    with open(f'D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\processed\{eid}_data_trimI_{condition}{agr}_{average}','wb') as f:
        pkl.dump(data_trimI, f)

def pid_stim_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False, average=False, discretize=True, method='all'):

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
    neural_data, regions = combine_probes(spike_count_stim_probe0, spike_count_stim_probe1, regions_probe0, regions_probe1, aggregate=aggregate, average=average)
    if discretize:
        neural_data = discretize_neural_data(neural_data, method=method)
    plot_neurons(neural_data, regions, eid, 'stim', aggregate)
    data_unique, data_synergy, data_redundancy, data_neuron, data_coninfo, data_trimI = run_pid_for_sources(decoding_variable, neural_data, regions)
    # mutual_information_data = run_mutual_information(decoding_variable, neural_data, regions)
    write_to_disk(eid, data_unique, data_synergy, data_redundancy, data_neuron, data_coninfo, data_trimI,  'stim', aggregate)

def pid_choice_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False, average=False, discretize=True, method='all'):

    (
        spike_count_choice_probe0, 
        spike_count_choice_probe1, 
        cluster_id_probe0, 
        cluster_id_probe1, 
        decoding_variable
    ) = gather_data_choice(trials_df, spikes_probe0, spikes_probe1)

    regions_probe0 = clusters_probe0['Beryl'][cluster_id_probe0].to_numpy()
    regions_probe1 = clusters_probe1['Beryl'][cluster_id_probe1].to_numpy()

    # drop unwanted neurons
    neural_data, regions = combine_probes(spike_count_choice_probe0, spike_count_choice_probe1, regions_probe0, regions_probe1, aggregate=aggregate, average=average)
    if discretize:
        neural_data = discretize_neural_data(neural_data, method=method)
    plot_neurons(neural_data, regions, eid, 'choice', aggregate)
    data_unique, data_synergy, data_redundancy, data_neuron, data_coninfo, data_trimI = run_pid_for_sources(decoding_variable, neural_data, regions)
    # mutual_information_data = run_mutual_information(decoding_variable, neural_data, regions)
    write_to_disk(eid, data_unique, data_synergy, data_redundancy, data_neuron, data_coninfo, data_trimI, 'choice', aggregate)


def pid_feedback_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False, average=False, discretize=True, method='all'):

    (
        spike_count_feedback_probe0, 
        spike_count_feedback_probe1, 
        cluster_id_probe0, 
        cluster_id_probe1, 
        decoding_variable
    ) = gather_data_feedback(trials_df, spikes_probe0, spikes_probe1)

    regions_probe0 = clusters_probe0['Beryl'][cluster_id_probe0].to_numpy()
    regions_probe1 = clusters_probe1['Beryl'][cluster_id_probe1].to_numpy()

    # drop unwanted neurons
    neural_data, regions = combine_probes(spike_count_feedback_probe0, spike_count_feedback_probe1, regions_probe0, regions_probe1, aggregate=aggregate, average=average)
    if discretize:
        neural_data = discretize_neural_data(neural_data, method=method)
    # plot_neurons(neural_data, regions, eid, 'feedback', aggregate)
    data_unique, data_synergy, data_redundancy, data_neuron, data_coninfo, data_trimI = run_pid_for_sources(decoding_variable, neural_data, regions)
    # mutual_information_data = run_mutual_information(decoding_variable, neural_data, regions)
    write_to_disk(eid, data_unique, data_synergy, data_redundancy, data_neuron, data_coninfo, data_trimI, 'feedback', aggregate)

def pid_prior_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False, average=False, discretize=True, method='all',prior='action-kernel'):

    (
        spike_count_prior_probe0, 
        spike_count_prior_probe1, 
        cluster_id_probe0, 
        cluster_id_probe1, 
        decoding_variable
    ) = gather_data_prior(trials_df, spikes_probe0, spikes_probe1, prior=prior)

    regions_probe0 = clusters_probe0['Beryl'][cluster_id_probe0].to_numpy()
    regions_probe1 = clusters_probe1['Beryl'][cluster_id_probe1].to_numpy()

    # drop unwanted neurons
    neural_data, regions = combine_probes(spike_count_prior_probe0, spike_count_prior_probe1, regions_probe0, regions_probe1, aggregate=aggregate, average=average)
    if discretize:
        neural_data = discretize_neural_data(neural_data, method=method)

    data_unique, data_synergy, data_redundancy, data_neuron, data_coninfo, data_trimI = run_pid_for_sources(decoding_variable, neural_data, regions)
    # mutual_information_data = run_mutual_information(decoding_variable, neural_data, regions)
    write_to_disk(eid, data_unique, data_synergy, data_redundancy, data_neuron, data_coninfo, data_trimI, prior ,aggregate, average)


def load_action_kernel(eid):
    location = f'D:\\personal\\phD\\code\\information-decomposition\\ibl-partial-info-decomp\\data\\raw\\{eid}_actionkernel_prior.csv'
    df = pd.read_csv(location)
    return df['prior']



def run_subsampling_analysis(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False, average=False,  discretize=True, method='all'):

    # run subsampling only on choice data
    (
        spike_count_choice_probe0, 
        spike_count_choice_probe1, 
        cluster_id_probe0, 
        cluster_id_probe1, 
        decoding_variable
    ) = gather_data_choice(trials_df, spikes_probe0, spikes_probe1)

    regions_probe0 = clusters_probe0['Beryl'][cluster_id_probe0].to_numpy()
    regions_probe1 = clusters_probe1['Beryl'][cluster_id_probe1].to_numpy()

    # drop unwanted neurons
    neural_data, regions = combine_probes(spike_count_choice_probe0, spike_count_choice_probe1, regions_probe0, regions_probe1, aggregate=aggregate, average=average)
    if discretize:
        neural_data = discretize_neural_data(neural_data, method=method)

    # we subsample here essentially
    means_triplets = []
    means_unique = []
    repeats = 10
    for idx in range(repeats):
        subsampled_data,subsampled_decoding = subsample(neural_data, decoding_variable)    
        data_unique, data_synergy, data_redundancy, data_neuron, data_coninfo, data_trimI = run_pid_for_sources(subsampled_decoding, subsampled_data, regions)
        # just keep means_triplets for all the repetations
        if aggregate==False:
            mean_for_keys = []
            for k in data_synergy.keys():
                mean_for_keys.append([np.mean(data_synergy[k]), np.mean(data_redundancy[k]), np.mean(data_coninfo[k]), np.mean(data_trimI[k])])
            means_triplets.append(mean_for_keys)

            means_for_unq = []
            for k in data_unique.keys():
                means_for_unq.append([np.mean(data_unique[k]), np.mean(data_neuron[k])])
            means_unique.append(means_for_unq)
        else:
            means_triplets.append([list(data_synergy.values()), list(data_redundancy.values()), list(data_coninfo.values()), list(data_trimI.values())])
            # means_unique.append([list(data_unique.values()), list(data_neuron.values())])
            means_for_unq = []
            for k in data_unique.keys():
                means_for_unq.append([np.mean(data_unique[k]), np.mean(data_neuron[k])])
            means_unique.append(means_for_unq)
    
    means_triplets = np.asarray(means_triplets)
    means_unique = np.asarray(means_unique)
    
    # write to disk
    if aggregate==False:
        agr = ''
    else:
        agr = 'agr'
    
    if discretize==False:
        dis = ''
        method = ''
    else:
        dis = 'dis'

    with open(f'D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\processed\{eid}_subsampled_trivariate_{agr}_{dis}_{method}.pkl','wb') as f:
        pkl.dump(means_triplets, f)

    with open(f'D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\processed\{eid}_subsampled_biivariate_{agr}_{dis}_{method}.pkl','wb') as f:
        pkl.dump(means_unique, f)
    

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

    if len(pids)==1:
        print(f'single insertion: deal with this {eid}')
        return


    
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
    
    #load action kernel data
    prior = load_action_kernel(eid)
    trials_df['prior'] = prior
    trials_df['prior-binary'] = np.asarray(trials_df['prior']>=0.5, dtype=np.int16)

    #mask it
    #subset trials based on trials_mask
    trials_df = trials_df[trials_mask]

    K = 2
    trials_df_glm = load_state_dataframe(eid, K=K)
    trials_df_glm['state'] = trials_df_glm[f'glm-hmm_{K}'].apply(lambda x: np.argmax(x))

    #subset it
    trials_df_glm = trials_df_glm[trials_mask]


    #NOTE: Proper order of things is to discretize, maintain count, no averaging for single neurons
    #NOTE: Same for aggregated regions, uniform discretization is good enough over each neuron
    # run partial information decompositions

    pid_stim_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False)
    pid_choice_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False)
    pid_feedback_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False)
    pid_prior_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False, prior='action-kernel')
    pid_prior_data(eid, trials_df_glm, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False, prior='glm-hmm')

    # pid_stim_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=True, average=True)
    # pid_choice_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=True, average=True)
    # pid_feedback_data(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=True, average=True)

    # run subsampling analysis for stability
    # only for choice, both for regions, aggregate and discretize
    # run_subsampling_analysis(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False, average=False,  discretize=True)
    # run_subsampling_analysis(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=True, average=False,  discretize=True)
    
    # run_subsampling_analysis(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False, average=False,  discretize=True, method='all')
    # run_subsampling_analysis(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=True, average=False,  discretize=True, method='all')

    # run_subsampling_analysis(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=False, average=False,  discretize=True, method='neuron')
    # run_subsampling_analysis(eid, trials_df, spikes_probe0, spikes_probe1, clusters_probe0, clusters_probe1, aggregate=True, average=False,  discretize=True, method='neuron')
    


if __name__=='__main__':

    # eids_df = pd.read_csv('D:\personal\phD\code\information-decomposition\ibl-partial-info-decomp\data\processed\eids_to_analyse.csv')
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international")
    # eids = eids_df.eid

    eids = np.load('D:\\personal\\phD\\code\\information-decomposition\\ibl-partial-info-decomp\\data\\processed\\eids_with_detailed_insertions.npy',allow_pickle=True)

    # run eid 0 and 3 (I think)
    #run_single_eid(one, eids[0])
    for idx in range(1, len(eids)):
        run_single_eid(one, eids[idx])
