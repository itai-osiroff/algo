function origins_LF=calc_back(LF, Na, mask_s, emission_z, x_stp_smp, x_max, lf_ang_grid, u_x_max)
% Create the pixel grid for the illumination plane
g = -u_x_max:x_stp_smp:u_x_max;
[g1, g2] = ndgrid(g);

% Iterate through the viewpoints to calculate the original positions of each
% point in the LF back to the original image using the slopes
for i1 = 1:Na
    for i2 = 1:Na

        % Calculate the ray direction with normaliztion to 1
        r1 = lf_ang_grid(i1);
        r2 = lf_ang_grid(i2);
        r3 = sqrt(1-r1^2-r2^2);

        % Calculate scale according to the distance between the
        % image plane and the phase mask
        scl = emission_z/r3;

        % Claculate the locations on the phase mask from which came the rays
        disp1 = r1 * scl;
        disp2 = r2 * scl;
        new_p1 = g1 + disp1;
        new_p2 = g2 + disp2;
        
        % Convert to pixel indices
        ind_new_p1 = round((new_p1 + x_max)/x_stp_smp) + 1;
        ind_new_p2 = round((new_p2 + x_max)/x_stp_smp) + 1;
        
        % Calculate new ray direction
        n1_mask = size(mask_s,1);
        ind_new_p1_1d = ind_new_p1 + (ind_new_p2-1) * n1_mask;
        new_r1 = r1 + mask_s(ind_new_p1_1d);
        ind_new_p2_1d = ind_new_p1 + (ind_new_p2-1) * n1_mask + n1_mask^2;
        new_r2 = r2 + mask_s(ind_new_p2_1d);
        new_norm = max(sqrt(new_r1.^2+new_r2.^2), 10^-5);
        mask_invalid = new_norm < 1;
        new_r1 = new_r1./max(1, new_norm);
        new_r2 = new_r2./max(1, new_norm);
        new_r3 = sqrt(1-min(new_r1.^2+new_r2.^2, 1));
        
        % And the new scale factor
        new_scl = emission_z./new_r3;

        % Repeat calculation to find the original location on the
        % illumination plane
        new_p1_img = g1 + disp1 - new_scl.*new_r1;
        new_p2_img = g2 + disp2 - new_scl.*new_r2;
        new_p1_img_ind = round((new_p1_img + u_x_max)/x_stp_smp) + 1;
        new_p2_img_ind = round((new_p2_img + u_x_max)/x_stp_smp) + 1;
   
        origins_LF(:, :, i1 ,i2) = interp2D_frw(LF(:, :, i1, i2).*mask_invalid, new_p1_img_ind, new_p2_img_ind);
        
    end 
end
