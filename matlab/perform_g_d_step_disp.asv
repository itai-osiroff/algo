function [updated_mask, update_step, cost] = perform_g_d_step_disp(LF, disp_mask, Na, x_stp_smp, u_x_max, cost_mask, k_num, k_limit, gd_step, g_filt_var, thresh_cof)

% which step_sizes to check
k = gpuArray(linspace(-k_limit, k_limit, k_num)); % k vector

% initalize gradient and cost matrices
grad_mat = zeros([size(disp_mask), k_num], 'gpuArray');
cost_vec = zeros([k_num, 1], 'gpuArray');

% Smoothing mask for the gradients
% g_filt_var = 10; % variance of the gaussian filter
g_filt = gpuArray(exp(-0.5/g_filt_var^2*(-round(3*g_filt_var):round(3*g_filt_var)).^2)); % Gaussian filter, zero mean and set variance
g_filt = g_filt./sum(g_filt);

% Iterate through the different gradient step_sizes
for i = 1:k_num
    [grad_mat(:, :, :, :, :, i), ~, cost_vec(i), ~] = calc_grad_disp(LF, Na, disp_mask, x_stp_smp, u_x_max, cost_mask, k(i), false);
    
    for ax = 1:2
        for i1 = 1:Na
            for i2 = 1:Na
                aux_ones = ones(size(grad_mat, 1), size(grad_mat, 2), "gpuArray");
                aux_ones = conv2(g_filt, g_filt', aux_ones, 'same');
                grad_mat(:, :, i1, i2, ax, i) = conv2(g_filt, g_filt', grad_mat(:, :, i1, i2, ax, i), 'same')./aux_ones; % Smooth each gradient per axis per k value
            end
        end
    end
end

% Per k value per pixel, take the value of the gradient corresponding to
% the highest cost
updated_mask = disp_mask;
[~, init_cost, ~, ~] = calc_grad_disp(LF, Na, updated_mask, x_stp_smp, u_x_max, cost_mask, 0, false);
[max_vals, max_inds] = max(grad_mat, [], 6);
% thresh_cof = 0.2;
thresh = max_vals > thresh_cof*mean(grad_mat.*(grad_mat>0), "all"); %Zero out the change for pixels which do not change the cost
update_step = k(max_inds).*thresh;
% aux_ones = ones(size(update_step), "gpuArray");
aux_ones = thresh;
aux_ones = convn(convn(aux_ones, g_filt, 'same'), g_filt', "same");
%aux_ones_inds = find(aux_ones); %avoid dividing by zero
update_step = convn(convn(update_step, g_filt, 'same'), g_filt', "same");
update_step = update_step./max(10^-3, aux_ones);

% Check smaller gd step_sizes to maximize score
for i = 1:4
    updated_mask_try = updated_mask + gd_step*update_step;
    [~, cost_try, ~, ~] = calc_grad_disp(LF, Na, updated_mask_try, x_stp_smp, u_x_max, cost_mask, 0, false);
    if cost_try < init_cost
        gd_step = gd_step / 2;
    else
        updated_mask = updated_mask_try;
        break
    end
end

%return the cost of the final updated_mask
[~, cost, ~, ~] = calc_grad_disp(LF, Na, updated_mask, x_stp_smp, u_x_max, cost_mask, 0, false);
% cost = max(init_cost, cost_try);

% if i == 3
%     updated_mask = updated_mask - gd_step*update_step;
% end

% updated_mask = updated_mask - gd_step*update_step;

% Display the corrected LF according to the updated mask
% [~, ~, ~, ~] = calc_grad_disp(LF, Na, updated_mask, x_stp_smp, u_x_max, cost_mask, 0, true);