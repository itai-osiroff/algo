import h5py
mat_contents = h5py.File(r"C:\Users\user\OneDrive - Technion\office computer files\images_output\marina_2024\2024_03_05_10_18_10_chicken_light_field_BSI_4beads.mat")
views_matlab = np.array(mat_contents["Images_apt"])
views_matlab = np.reshape(views_matlab, (5, 5, 220, 220))
views = views_matlab
view_grid_size = 5