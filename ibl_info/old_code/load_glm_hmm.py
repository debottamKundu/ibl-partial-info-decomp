
import pandas as pd
import numpy as np
from glm_hmm.predict_sessions import run_glm_for_session
from pathlib import Path

from one.api import ONE


def load_state_dataframe(eid, K):
    """
    Returns dataframe with glm-hmm states 

    Args:
        eid (string): session id
        K (int): GLM states

    Returns:
        df: Dataframe containing glm-hmm fit
    """
    
    one = ONE(base_url='https://alyx.internationalbrainlab.org/')
    data_dir = Path('D:\personal\phD\code\GLM-HMM\glm_hmm\outputs')
    output_dir = data_dir/'individual_sessions'
    subject = Path(one.get_details(eid)['subject'])

    if Path.exists(output_dir/subject/eid):
        filename = output_dir/subject/eid/'trials.pqt'
        df = pd.read_parquet(filename)

        # now check if column with particular k exists
        if f'glm-hmm_{K}' in df.columns:
            return df
        else:
            df = run_glm_for_session(eid, K)
    else:
        df = run_glm_for_session(eid, K)
                      
    return df


def disengaged_engaged_values(eid, K=2):

    df = load_state_dataframe(eid, K)
    biased_df = df[~(df.probabilityLeft==0.5)]
    glm_fit = np.concatenate(biased_df[f"glm-hmm_{K}"].values).reshape(-1, K)

    states = np.zeros((glm_fit.shape[0], K))
    for idx in range(K):
        states[:, idx] = np.argmax(glm_fit, axis=1) == idx
    
    return states