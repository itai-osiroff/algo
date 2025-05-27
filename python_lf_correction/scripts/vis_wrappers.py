#This script holds functions which are used to visualize LF data

import numpy as np
import matplotlib.pyplot as plt

def displayLF(LF, border=0):
    """
    Display a light-field capture as a grid of sub_images.
    Args:
        LF (numpy.ndarray): Light-field data with shape (H, W, nx, ny).
        border (int): Width of the border between sub-images.

    Returns:
        None: The function simply opens a display window showing the light-field.
    """

    LF = np.pad(LF, ((border, border), (border, border), (0, 0), (0, 0)), mode='constant', constant_values=0)
    H, W, nx, ny = LF.shape

    LFf = np.zeros((H*ny, W*nx), dtype=LF.dtype)
    for i in range(nx):
        for j in range(ny):
            LFf[j*H:(j+1)*H, i*W:(i+1)*W] = LF[:, :, i, j]
    
    plt.imshow(LFf, cmap='gray')