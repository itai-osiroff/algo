# This script holds functions which are meant to load data from .mat files, convert them to numpy objects and return them for use

import numpy as np
import scipy.io as sio
import h5py

def load_mat_file(file_path):
    """
    Load a .mat file and return its contents as a dictionary.
    
    Args:
        file_path (str): Path to the .mat file.
        
    Returns:
        dict: Dictionary containing the contents of the .mat file.
    """
    try:
        mat_contents = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        print("scipy.io.loadmat does not support 7.3 matlab files, trying h5py instead.")
        mat_contents = h5py.File(file_path, 'r')
        mat_contents = {key: np.array(value) for key, value in mat_contents.items() if not key.startswith('__')}
    return mat_contents