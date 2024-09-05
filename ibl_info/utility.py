# utility functions
# - aggregating clusters into regions
# - helpers for plotting results
# - computing PID for a target and 2 sources
# - computing net synergy/redundancy for one target and two sources
# - computing just MI 


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
from idtxl.data import Data
from idtxl.bivariate_pid import BivariatePID
from idtxl.bivariate_mi import BivariateMI
from idtxl.estimators_jidt import JidtDiscreteMI
from sklearn.metrics import mutual_info_score

def aggregated_regions_time_resolved(binned_spike_counts, cluster_acronyms):
    """generates summed over spike counts for each region, with a time resolution provided beforehand

    Args:
        binned_spike_counts (np.array): trials x neurons x timepoints
        cluster_acronyms (np.array): neurons

    Returns:
        np.array: binned counts aggregated by regions, and names of regions
    """
    # binned_spike_counts = trials x neurons x timepoints

    regions = np.unique(cluster_acronyms)
    data = np.zeros(
        (binned_spike_counts.shape[0], len(regions), binned_spike_counts.shape[-1])
    )  # trials, regions, timepoints
    for idx, r in enumerate(regions):
        neurons = np.argwhere(cluster_acronyms == r).reshape(
            -1,
        )
        aggregate_cluster = np.sum(binned_spike_counts[:, neurons, :], axis=1)
        data[:, idx, :] = aggregate_cluster
    return data, regions


def aggregated_regions_time_intervals(spike_counts, cluster_acronyms):
    """generates total number of spikes per region for different trials

    Args:
        spike_counts (np.array): spike count for each trial from each neuron, # trials x neurons
        cluster_acronyms (np.array): neuron locations
    Returns:
        np.array: spike count aggregated by regions, and names of  corresponding regions
    """


    regions = np.unique(cluster_acronyms)
    data = np.zeros(
        (spike_counts.shape[0],len(regions)) # trials x regions
    )  
    for idx, r in enumerate(regions):
        neurons = np.argwhere(cluster_acronyms == r).reshape(
            -1,
        )
        aggregate_cluster = np.sum(spike_counts[:, neurons], axis=1)
        data[:, idx] = aggregate_cluster
    return data, regions

def computepid_time_intervals(Y, X1, X2, lags=[0, 0]):
    """compute broja pid for a single target with 2 sources

    Args:
        Y (np.array): target, normally a behavioral variable or a neuron
        X1 (np.array): source 1, binned spike counts for a neuron
        X2 (np.array): source 2, binned spike counts for another neuron
        lags (list, optional): Lags to consider. Defaults to [0,0].
    """
    data = np.vstack([Y, X1, X2])
    data = Data(data, dim_order="pr", normalise=False)
    settings_tartu = {"pid_estimator": "TartuPID", "lags_pid": lags}
    pid = BivariatePID()
    result = pid.analyse_single_target(data=data, settings=settings_tartu, target=0, sources=[1, 2])
    return [
        result.get_single_target(0)["unq_s1"],
        result.get_single_target(0)["unq_s2"],
        result.get_single_target(0)["shd_s1_s2"],
        result.get_single_target(0)["syn_s1_s2"],
    ]

def computepid_time_resolved(Y, X1, X2, lags=[3,3]):
    """Compute BROJA PID for a single target with two sources, in a time resolved fashion

    Args:
        Y (np.array): target, behavioral variable or neuron values over time
        X1 (np.array): source 1, trials x timepoints
        X2 (np.array): source 2, trials x timepoints
        lags (list, optional): Lag to consider Defaults to [3,3].
    """

    data = np.stack(Y, X1, X2)
    data = Data(data, dim_order = "prs", normalise=False) # processes, repetations, samples

    settings_tartu = {"pid_estimator": "TartuPID", "lags_pid": lags}
    pid = BivariatePID()
    result = pid.analyse_single_target(
        data=data, settings=settings_tartu, target=0, sources=[1, 2]
    )
    return [
        result.get_single_target(0)["unq_s1"],
        result.get_single_target(0)["unq_s2"],
        result.get_single_target(0)["shd_s1_s2"],
        result.get_single_target(0)["syn_s1_s2"],
    ]

def compute_netsynred(Y, X1, X2):
    """Compute MI(Y; X1, X2) - MI(Y,X1) - MI(Y,X2)  

    Args:
        Y (np.asarray): target, normally a behavioral variable or a neuron 
        X1 (np.asarray): source 1, binned spike counts for a neuron 
        X2 (np.asarray): source 2, binned spike counts for another neuron 
        lags (list, optional): Lags to consider. Defaults to [0, 0].
    """

    # compute bivariate information first
    data = np.vstack(Y, X1, X2)
    data = Data(data, dim_order="pr", normalise=False)
    settings_bivariate_mi = {'cmi_estimator': 'JidtGaussianCMI',
            'max_lag_sources': 0,
            'min_lag_sources': 0}
    
    settings_mi = {"discretise_method": "none"} 
    mi_estimator = JidtDiscreteMI(settings=settings_mi)
    miYX1 = mi_estimator.estimate(Y, X1)
    miYX2 = mi_estimator.estimate(Y, X2)

    bivariate = BivariateMI()
    result = bivariate.analyse_single_target(settings=settings_bivariate_mi, data=data, target=0, sources=[1,2]) 
    miYX1X2 = result.get_single_target(0, fdr=False)['mi'][0]

    syn_red_index = miYX1X2 - miYX1 - miYX2

    # if syn_red>0 then more synergy otherwise more redundancy
    return syn_red_index


def compute_BivariateMI(Y, X1, X2):
    """Compute MI(Y; X1, X2)  

    Args:
        Y (np.asarray): target, normally a behavioral variable or a neuron 
        X1 (np.asarray): source 1, binned spike counts for a neuron 
        X2 (np.asarray): source 2, binned spike counts for another neuron 
    """

    # compute bivariate information first
    data = np.vstack([Y, X1, X2])
    data = Data(data, dim_order="pr", normalise=False)
    settings_bivariate_mi = {'cmi_estimator': 'JidtGaussianCMI',
            'max_lag_sources': 0,
            'min_lag_sources': 0}
    
    bivariate = BivariateMI()
    result = bivariate.analyse_single_target(settings=settings_bivariate_mi, data=data, target=0, sources=[1,2]) 
    miYX1X2 = result.get_single_target(0, fdr=False)['mi'][0]

    return miYX1X2