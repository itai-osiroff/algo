import numpy as np
import sys
sys.path.insert(0, './python_lf_correction/lib')

from matlab_data_load import load_mat_file
from vis_wrappers import displayLF
import matlab.engine


data = load_mat_file(r"matlab_files\example_LF.mat")
mask = data["mask_s"]
displayLF(mask, border=0)
