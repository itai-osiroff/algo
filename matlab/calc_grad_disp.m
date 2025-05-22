function [grad, cost_0, cost_post, LF_frw]=calc_grad_disp(LF, Na, disp_mask, x_stp_smp, u_x_max, cost_mask, delta, calc_LF_flag)
grad = zeros(size(disp_mask)); % This is a change
cost_0 = 0; % initialize zero cost
cost_1_1 = 0; % initialize post-grad cost, ax=1
cost_1_2 = 0; % initialize post-grad cost, ax=2
LF_frw = zeros(size(LF), "gpuArray");

% Create the pixel grid for the illumination plane
g = linspace(-u_x_max, u_x_max, size(LF, 1)); % This is a change
g = gpuArray(g);
[g1, g2] = ndgrid(g);

% Iterate through the viewpoints to calculate the original positions of each
% point in the LF back to the original image using the slopes
for i1 = 1:Na
    for i2 = 1:Na
        
        mask_invalid_0 = ones(size(LF, 1), size(LF, 2), "gpuArray");
        % Calculate the new coordinates in the image
        new_p1_img = g1 + disp_mask(:, :, i1, i2, 1);
        new_p2_img = g2 + disp_mask(:, :, i1, i2, 2);
        new_p1_img_ind = (new_p1_img + u_x_max)/x_stp_smp + 1;
        new_p2_img_ind = (new_p2_img + u_x_max)/x_stp_smp + 1;
        % mask_invalid_0 = (new_p1_img_ind >= 1).*(new_p1_img_ind <= size(LF, 1)).*(new_p2_img_ind >= 1).*(new_p2_img_ind <= size(LF, 2));
        % % new_p1_img_ind = new_p1_img_ind.*mask_invalid_0 + not(mask_invalid_0);
        % new_p2_img_ind = new_p2_img_ind.*mask_invalid_0 + not(mask_invalid_0);
    
        cost_warp_mask0 = interp2D_bkw(cost_mask.*mask_invalid_0, new_p1_img_ind, new_p2_img_ind).*LF(:, :, i1, i2);
        
        % cost_warp_mask1 = interp2(cost_mask.*mask_invalid_0, new_p1_img_ind, new_p2_img_ind,'linear').*LF(:, :, i1, i2);

        % if i1 == 3 && i2 == 3
        %     keyboard
        % end
        %calculate the score
        cost_0 = cost_0 + sum(sum(cost_warp_mask0));
        mask_invalid_1 = ones(size(LF, 1), size(LF, 2), 2, "gpuArray");
        per_disp_mask = zeros(size(disp_mask), "gpuArray");
        for ax = 1:2 % iterate per component: 1 = y-axis, 2 = x-axis
            % Implement delta perturbation
            if ax == 1
                per_disp_mask(:, :, i1, i2, 2) = disp_mask(:, :, i1, i2, 2);
                per_disp_mask(:, :, i1, i2, 1) = disp_mask(:, :, i1, i2, 1) + delta;
            else
                per_disp_mask(:, :, i1, i2, 1) = disp_mask(:, :, i1, i2, 1);
                per_disp_mask(:, :, i1, i2, 2) = disp_mask(:, :, i1, i2, 2) + delta;
            end
            
            % Repeat calculation to find the original location on the
            % illumination plane
            new_p1_img = g1 + per_disp_mask(:, :, i1, i2, 1);
            new_p2_img = g2 + per_disp_mask(:, :, i1, i2, 2);
            new_p1_img_ind = (new_p1_img + u_x_max)/x_stp_smp + 1;
            new_p2_img_ind = (new_p2_img + u_x_max)/x_stp_smp + 1;
            % mask_invalid_1(:, :, ax) = (new_p1_img_ind >= 1).*(new_p1_img_ind <= size(LF, 1)).*(new_p2_img_ind >= 1).*(new_p2_img_ind <= size(LF, 2));
            % new_p1_img_ind = new_p1_img_ind.*mask_invalid_1(: ,:, ax) + not(mask_invalid_1(:, :, ax));
            % new_p2_img_ind = new_p2_img_ind.*mask_invalid_1(:, :, ax) + not(mask_invalid_1(:, :, ax));
            cost_warp_mask1 = interp2D_bkw(cost_mask.*mask_invalid_1(: ,:, ax), new_p1_img_ind, new_p2_img_ind).*LF(:, :, i1, i2);
        
            % if i1 == 6 && i2 == 2
            %     figure;imagesc(cost_warp_mask1); colorbar;
            %     keyboard
            % end
            t_grad = (cost_warp_mask1 - cost_warp_mask0).*min(mask_invalid_0, mask_invalid_1(:, :, ax));
            if ax == 1
                grad(:, :, i1 ,i2, 1) = grad(:, :, i1 ,i2, 1) + t_grad;
                cost_1_1 = cost_1_1 + sum(sum(cost_warp_mask1));
            else
                % if i1 ==4 && i2==4
                %     keyboard
                % end
                grad(:, :, i1 ,i2, 2) = grad(:, :, i1 ,i2, 2) + t_grad;
                cost_1_2 = cost_1_2 + sum(sum(cost_warp_mask1));
            end
        end
        % if i1 ==4 && i2==2
        %     keyboard
        % end
        if calc_LF_flag
            LF_frw(:, :, i1, i2) = interp2D_frw(LF(:, :, i1, i2), new_p1_img_ind, new_p2_img_ind);
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