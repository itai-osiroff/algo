function grad = calc_num_grad(LF, Na, Nxw, smp_stp, mask_s, emission_z, x_stp_smp, x_max, lf_ang_grid, u_x_max, cost_mask, delta)
grad = zeros((Nxw+smp_stp-1)/smp_stp, (Nxw+smp_stp-1)/smp_stp, 2, 'gpuArray'); % Initialize zero gradient

% Create the pixel grid for the illumination plane
g = -u_x_max:x_stp_smp:u_x_max;
[g1, g2] = ndgrid(g);

cost_0 = 0; %initialize original cost
% Calculate the initial cost
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
        new_r3 = sqrt(1-new_r1.^2-new_r2.^2);
        
        % And the new scale factor
        new_scl = emission_z./new_r3;

        % Repeat calculation to find the original location on the
        % illumination plane
        new_p1_img = g1 + disp1 - new_scl.*new_r1;
        new_p2_img = g2 + disp2 - new_scl.*new_r2;
        new_p1_img_ind = (new_p1_img + u_x_max)/x_stp_smp + 1;
        new_p2_img_ind = (new_p2_img + u_x_max)/x_stp_smp + 1;
   
        %origins_LF(:, :, i1 ,i2) = interp2D_frw(LF(:, :, i1, i2), new_p1_img_ind, new_p2_img_ind);
        cost_warp_mask0 = interp2D_bkw(cost_mask, new_p1_img_ind, new_p2_img_ind).*LF(:, :, i1, i2);
        %calculate the initial cost
        cost_0 = cost_0 + sum(sum(cost_warp_mask0));
    end
end



% Iterate through the entire phase mask (mask_s), pixels, viewpoints,
% x-y-components, and calculate the gradient of the cost with respect
% to each one
for j1 = 1:length(grad)
    for j2 = 1:length(grad)
        for ax = 1:2
            % Randomly sample pixels of the gradient
            % Sample heavily in the center
            if abs(j1 - length(grad)/2) <= 20 && abs(j2 - length(grad)/2) <= 20
                if random("uniform", 1, 2) < (1+1/3)
                    continue
                end
            else % Sample more sparsly closr to the edges
                if random("uniform", 1, 2) < (1.9)
                    continue
                end
            end
            cost_1 = 0; %initialize cost for perturbated mask
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
                    
                    % Implement delta perturbation in the right cell
                    per_mask_s = mask_s;
                    per_mask_s(j1, j2, ax) = mask_s(j1, j2, ax) + delta;
        
                    % Calculate new ray direction
                    n1_mask = size(per_mask_s,1);
                    ind_new_p1_1d = ind_new_p1 + (ind_new_p2-1) * n1_mask;
                    new_r1 = r1 + per_mask_s(ind_new_p1_1d);
                    ind_new_p2_1d = ind_new_p1 + (ind_new_p2-1) * n1_mask + n1_mask^2;
                    new_r2 = r2 + per_mask_s(ind_new_p2_1d);
                    new_r3 = sqrt(1-new_r1.^2-new_r2.^2);
                    
                    % And the new scale factor
                    new_scl = emission_z./new_r3;
            
                    % Repeat calculation to find the original location on the
                    % illumination plane
                    new_p1_img = g1 + disp1 - new_scl.*new_r1;
                    new_p2_img = g2 + disp2 - new_scl.*new_r2;
                    new_p1_img_ind = (new_p1_img + u_x_max)/x_stp_smp + 1;
                    new_p2_img_ind = (new_p2_img + u_x_max)/x_stp_smp + 1;
               
                    %origins_LF(:, :, i1 ,i2) = interp2D_frw(LF(:, :, i1, i2), new_p1_img_ind, new_p2_img_ind);
                    cost_warp_mask1 = interp2D_bkw(cost_mask, new_p1_img_ind, new_p2_img_ind).*LF(:, :, i1, i2);
                    cost_1 = cost_1 + sum(sum(cost_warp_mask1));
                end
            end

            % Calculate the numerical gradient and allocate to cell
            grad(j1, j2, ax) = (cost_1-cost_0)/delta;
        end
    end
end
