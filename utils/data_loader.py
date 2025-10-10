"""
Utilities for loading and organizing PASTOR data
"""

import json
import numpy as np
from collections import defaultdict

import sys
sys.path.append('..')
from config.config import PASTORS, VALID_AAS


def load_pastor_data(json_path):
    """
    Load and organize PASTOR data from JSON file
    
    Parameters:
    -----------
    json_path : str
        Path to JSON data file
    
    Returns:
    --------
    pastor_groups : dict
        Organized PASTOR groups by channel
    aa_info : dict
        Amino acid information
    raw_data : dict
        Raw signal data
    channels : dict
        Channel information
    run : dict
        Run information
    """
    
    with open(json_path, "r") as file:
        data = json.load(file)
    
    aa_info = data.get("aa", {})
    raw_data = data.get("cleaned_segment", {})
    channels = data.get("channel", {})
    run = data.get("run", {})
    
    # Filter valid amino acids
    valid_indices = []
    valid_channels = []
    aas_list = []
    
    for key in aa_info.keys():
        aa_value = aa_info[key]
        if aa_value is not None and aa_value in VALID_AAS:
            valid_indices.append(key)
            valid_channels.append(channels[key])
            aas_list.append(aa_value)
    
    aas_list = ''.join(aas_list)
    
    # Build PASTOR groups
    pastor_groups = {pastor: defaultdict(list) for pastor in PASTORS}
    for i in range(len(aas_list) - 4):
        aa_slice = aas_list[i:i+5]
        if aa_slice in PASTORS:
            channels_slice = valid_channels[i:i+5]
            indices_slice = valid_indices[i:i+5]
            pastor_groups[aa_slice][int(channels_slice[0])].append(indices_slice)
    
    return pastor_groups, aa_info, raw_data, channels, run