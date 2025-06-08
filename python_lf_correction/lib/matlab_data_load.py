# This script holds functions which are meant to load data from .mat files, convert them to numpy objects and return them for use

import numpy as np
import scipy.io as sio


def load_mat_file(file_path):
    """
    Load a .mat file and return its contents as a dictionary.
    
    Args:
        file_path (str): Path to the .mat file.
        
    Returns:
        dict: Dictionary containing the contents of the .mat file.
    """
    mat_contents = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    return mat_contents