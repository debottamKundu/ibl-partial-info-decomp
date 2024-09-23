import pandas as pd
import numpy as pd
from pathlib import Path



def get_region(acronym):
    """
    Maps from acronym to region based on Harris et. al

    Args:
        acronym (string): IBL acronym
    Returns:
        str: Region name
        int: Hierarchy value
    """

    location = Path('../data/processed/region_mapping.csv')
    df = pd.read_csv(location)
    
    if df.areas.isin([acronym]).any():
        # if acronym in region

        region = df[df['areas']==acronym]['Region'].values.reshape(-1,)[0]
        hierarchy_ordinal = df[df['areas']==acronym]['H-value'].values.reshape(-1,)[0]
        return region, hierarchy_ordinal
    else:
        return None, None
    

def global_hierarchy(acronym):
    """
    Returns global hiearchy value

    Args:
        acronym (string): region/cluster name

    Returns:
        h_value (float): real valued hierarchy score
        rank (int): oridnal rank, with higher values meaning higher hierarchy
    """
    location = Path('../data/processed/global_hierarchy.csv')
    df = pd.read_csv(location)

    if df.areas.isin([acronym]).any():
        h_value = df[df['areas']==acronym]['H-score'].values.reshape(-1,)[0]
        rank = df[df['areas']==acronym]['ordinal_rank'].values.reshape(-1,)[0]

        return h_value, rank
    else:
        return None, None