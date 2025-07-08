import numpy as np
from idtxl.data import Data
from idtxl.bivariate_pid import BivariatePID
from idtxl.bivariate_mi import BivariateMI
from idtxl.estimators_jidt import JidtDiscreteMI


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
    
    settings_mi = {"discretise_method": "equal"} 
    mi_estimator = JidtDiscreteMI(settings=settings_mi)
    miYX1 = mi_estimator.estimate(Y, X1)
    miYX2 = mi_estimator.estimate(Y, X2)

    bivariate = BivariateMI()
    result = bivariate.analyse_single_target(settings=settings_bivariate_mi, data=data, target=0, sources=[1,2]) 
    miYX1X2 = result.get_single_target(0, fdr=False)['mi'][0]

    syn_red_index = miYX1X2 - miYX1 - miYX2

    # if syn_red>0 then more synergy otherwise more redundancy
    return syn_red_index

def compute_mutual_information(Y, X):
    """
    compute MI(Y,X)

    Args:
        Y (np.array): variable 1
        X (np.array): variable 2
    """
    settings_mi = {"discretise_method": "equal"} 
    mi_estimator = JidtDiscreteMI(settings=settings_mi)
    try:
        mi = mi_estimator.estimate(Y, X)
    except Exception as e:
        print(e)
        print(X)
        mi = 0
    return mi


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


def run_mutual_information(decoding_variable, neural_data, regions):
    # neural data is neurons x trials
    # regions is region_acronym

    MI_data = np.zeros((neural_data.shape[0]))
    for idx in tqdm(range(len(MI_data))):
        Y = np.asarray(decoding_variable, dtype=np.int32)
        X = np.asarray(neural_data[idx,:], dtype=np.int32)
        mi = compute_mutual_information(Y, X)

        MI_data[idx] = mi

    # now we build an dict
    data_mi = {}
    for idx in range(len(MI_data)):
        region = regions[idx]
        if region in data_mi.keys():
            temp = data_mi[region]
            temp.append(MI_data[idx])
            data_mi[region] = temp
        else:
            temp = [MI_data[idx]]
            data_mi[region] = temp
    return data_mi