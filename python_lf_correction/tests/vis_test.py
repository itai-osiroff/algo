import numpy as np
import sys
sys.path.insert(0, './python_lf_correction/scripts')

from matlab_data_load import load_mat_file
from vis_wrappers import displayLF

data = load_mat_file("matlab\example_LF.mat")
mask = data["mask"]
