function [grad, cost_0, cost_post, LF_frw]=calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, mask_s, emission_z, x_stp_smp, x_max, lf_ang_grid, u_x_max, cost_mask, delta, calc_LF_flag)
% grad = zeros((Nxw+smp_stp-1)/smp_stp, (Nxw+smp_stp-1)/smp_stp, 2, 'gpuArray'); % Initialize zero gradient
% grad = zeros((Nxw+smp_stp-1)/smp_stp, (Nxw+smp_stp-1)/smp_stp, 2); % This is a change
grad = zeros(size(mask_s)); % This is a change
cost_0 = 0; % initialize zero cost
cost_1_1 = 0; % initialize post-grad cost, ax=1
cost_1_2 = 0; % initialize post-grad cost, ax=2
LF_frw = zeros(size(LF), "gpuArray");

% Create the pixel grid for the illumination plane
% g = -u_x_max:x_stp_smp:u_x_max;
g = linspace(-u_x_max, u_x_max, size(LF, 1)); % This is a change
g = gpuArray(g);
[g1, g2] = ndgrid(g);

% Iterate through the viewpoints to calculate the original positions of each
% point in the LF back to the original image using the slopes
for i1 = 1:Na
    for i2 = 1:Na
        if i1 ~= 3 || i2 ~= 3
            continue
        end
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
        ind_new_p1 = round((new_p1 + x_max)/(x_stp_smp*stp_ratio)) + 1;
        ind_new_p2 = round((new_p2 + x_max)/(x_stp_smp*stp_ratio)) + 1;
        
        % Calculate new ray direction
        n1_mask = size(mask_s,1);
        ind_new_p1_1d = ind_new_p1 + (ind_new_p2-1) * n1_mask;
        ind_new_p2_1d = ind_new_p1 + (ind_new_p2-1) * n1_mask + n1_mask^2;
        mask_invalid_0 = (ind_new_p1_1d<=n1_mask^2*size(mask_s, 3)) .* (ind_new_p2_1d<=n1_mask^2*size(mask_s, 3)); %this is a change
        mask_invalid_0 = mask_invalid_0 .* (1<=ind_new_p1_1d) .* (1<=ind_new_p2_1d); %this is a change
        % ind_new_p1_1d = ind_new_p1_1d.*mask_invalid_0;
        % ind_new_p2_1d = ind_new_p2_1d.*mask_invalid_0;
        ind_new_p1_1d = ind_new_p1_1d.*mask_invalid_0 + not(mask_invalid_0); %this is a change
        ind_new_p2_1d = ind_new_p2_1d.*mask_invalid_0 + not(mask_invalid_0); %this is a change
        new_r1 = r1 + mask_s(ind_new_p1_1d).*mask_invalid_0;
        new_r2 = r2 + mask_s(ind_new_p2_1d).*mask_invalid_0;
        % new_r1 = zeros(size(g1));
        % new_r2 = zeros(size(g2));
        % new_r1(ind_new_p1_1d(mask_invalid_0)) = r1 + mask_s(ind_new_p1_1d(mask_invalid_0));
        % new_r2(ind_new_p1_1d(mask_invalid_0)) = r2 + mask_s(ind_new_p2_1d(mask_invalid_0));
        new_norm = max(sqrt(new_r1.^2+new_r2.^2), 10^-5);
        mask_invalid_0 = mask_invalid_0 .* new_norm < 1;
        new_r1 = new_r1./max(1, new_norm);
        new_r2 = new_r2./max(1, new_norm);
        new_r3 = sqrt(1-min(1, new_r1.^2+new_r2.^2));
        
        % And the new scale factor
        new_scl = emission_z./new_r3;

        % Repeat calculation to find the original location on the
        % illumination plane
        new_p1_img = g1 + disp1 - new_scl.*new_r1;
        new_p2_img = g2 + disp2 - new_scl.*new_r2;
        new_p1_img_ind = (new_p1_img + u_x_max)/x_stp_smp + 1;
        new_p2_img_ind = (new_p2_img + u_x_max)/x_stp_smp + 1;
        % original_new_p1_img = new_p1_img;
        % original_new_p2_img = new_p2_img;
    
        %origins_LF(:, :, i1 ,i2) = interp2D_frw(LF(:, :, i1, i2), new_p1_img_ind, new_p2_img_ind);
        cost_warp_mask0 = interp2D_bkw(cost_mask, new_p1_img_ind, new_p2_img_ind).*LF(:, :, i1, i2);
        % cost_warp_mask0 = interp2(cost_mask, new_p1_img_ind, new_p2_img_ind).*LF(:, :, i1, i2);
        % if i1 == 3 && i2 == 3
        %     keyboard
        % end
        %calculate the initial cost
        cost_0 = cost_0 + sum(sum(cost_warp_mask0));
        for ax = 1:2 % iterate per component: 1 = y-axis, 2 = x-axis
            % Implement delta perturbation in the right cell
            % per_mask_s = gather(mask_s);
            per_mask_s = mask_s;

            % [unq_idxs_1, locs_unq_idxs_1] = unique(ind_new_p1_1d);
            % [unq_idxs_2, locs_unq_idxs_2] = unique(ind_new_p2_1d);
            % if ax == 1
            %     per_mask_s(unq_idxs_1) = mask_s(locs_unq_idxs_1).*mask_invalid_0(locs_unq_idxs_1) + delta;
            % else
            %     per_mask_s(unq_idxs_2) = mask_s(locs_unq_idxs_2).*mask_invalid_0(locs_unq_idxs_2) + delta;
            % end
            if ax == 1
                per_mask_s(ind_new_p1_1d) = mask_s(ind_new_p1_1d).*mask_invalid_0 + delta;
            else
                % per_mask_s = gather(per_mask_s);
                per_mask_s(ind_new_p2_1d) = mask_s(ind_new_p2_1d).*mask_invalid_0 + delta;
            end
            % per_mask_s = gpuArray(per_mask_s);

            % Calculate new ray direction
            n1_mask = size(per_mask_s,1);
            ind_new_p1_1d = ind_new_p1 + (ind_new_p2-1) * n1_mask;
            ind_new_p2_1d = ind_new_p1 + (ind_new_p2-1) * n1_mask + n1_mask^2;
            mask_invalid_1 = (ind_new_p1_1d<=n1_mask^2*size(mask_s, 3)) .* (ind_new_p2_1d<=n1_mask^2*size(mask_s, 3));
            mask_invalid_1 = mask_invalid_1 .* (1<=ind_new_p1_1d) .* (1<=ind_new_p2_1d);
            % ind_new_p1_1d = ind_new_p1_1d.*mask_invalid_1;
            % ind_new_p2_1d = ind_new_p2_1d.*mask_invalid_1;
            ind_new_p1_1d = ind_new_p1_1d.*mask_invalid_1 + not(mask_invalid_1);
            ind_new_p2_1d = ind_new_p2_1d.*mask_invalid_1 + not(mask_invalid_1);
            new_r1 = r1 + per_mask_s(ind_new_p1_1d).*mask_invalid_1;
            new_r2 = r2 + per_mask_s(ind_new_p2_1d).*mask_invalid_1;
            % new_r1 = r1 + per_mask_s(ind_new_p1_1d);
            % new_r2 = r2 + per_mask_s(ind_new_p2_1d);
            new_norm = max(sqrt(new_r1.^2+new_r2.^2), 10^-5);
            mask_invalid_1(:, :, ax) = new_norm < 1;
            new_r1 = new_r1./max(1, new_norm);
            new_r2 = new_r2./max(1, new_norm);
            new_r3 = sqrt(1-min(1, new_r1.^2+new_r2.^2));
            
            % And the new scale factor
            new_scl = emission_z./new_r3;
    
            % Repeat calculation to find the original location on the
            % illumination plane
            new_p1_img = g1 + disp1 - new_scl.*new_r1;
            new_p2_img = g2 + disp2 - new_scl.*new_r2;
            new_p1_img_ind = (new_p1_img + u_x_max)/x_stp_smp + 1;
            new_p2_img_ind = (new_p2_img + u_x_max)/x_stp_smp + 1;
            cost_warp_mask1 = interp2D_bkw(cost_mask, new_p1_img_ind, new_p2_img_ind).*LF(:, :, i1, i2);
            % if i1 == 4 && i2 == 4
            %     keyboard
            % end
            t_grad = (cost_warp_mask1 - cost_warp_mask0).*min(mask_invalid_0, mask_invalid_1(: , :, ax));
            if ax == 1
                grad(ind_new_p1_1d) = grad(ind_new_p1_1d) + t_grad;
                cost_1_1 = cost_1_1 + sum(sum(cost_warp_mask1));
            else
                % if i1 ==4 && i2==4
                %     keyboard
                % end
                grad(ind_new_p2_1d) = grad(ind_new_p2_1d) + t_grad;
                cost_1_2 = cost_1_2 + sum(sum(cost_warp_mask1));
            end
        end
        if calc_LF_flag
            LF_frw(:, :, i1, i2) = interp2D_frw(LF(:, :, i1, i2).*min(mask_invalid_1(:, :, 1), mask_invalid_1(:, :, 2)), new_p1_img_ind, new_p2_img_ind);
        end
     end 
end
cost_post = cost_1_1 + cost_1_2;
grad = gpuArray(grad); % This is a change
if calc_LF_flag
    displayLF(LF_frw,1);
end

end
%grad = grad / delta;