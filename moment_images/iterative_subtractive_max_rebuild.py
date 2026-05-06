import scipy
new_reference_2 = np.zeros(global_average.shape)
pixel_range = 5
iter = 0
max_iter = 50
global_average_iter = global_average.copy()
while iter<max_iter:
    max_inds = np.unravel_index(np.argmax(global_average_iter), global_average_iter.shape)
    new_reference_2[max_inds[0]-pixel_range:max_inds[0]+pixel_range, max_inds[1]-pixel_range:max_inds[1]+pixel_range] = new_reference[max_inds[0]-pixel_range:max_inds[0]+pixel_range, max_inds[1]-pixel_range:max_inds[1]+pixel_range] + global_average_iter[max_inds[0]-pixel_range:max_inds[0]+pixel_range, max_inds[1]-5:max_inds[1]+pixel_range]
    global_average_iter[max_inds[0]-pixel_range:max_inds[0]+pixel_range, max_inds[1]-pixel_range:max_inds[1]+pixel_range] = 0
    # global_average_iter[max_inds[0]-pixel_range*2:max_inds[0]+pixel_range*2, max_inds[1]-pixel_range*2:max_inds[1]+pixel_range*2] = scipy.ndimage.gaussian_filter(global_average_iter[max_inds[0]-pixel_range*2:max_inds[0]+pixel_range*2, max_inds[1]-pixel_range*2:max_inds[1]+pixel_range*2], 2, truncate=2)
    iter = iter + 1