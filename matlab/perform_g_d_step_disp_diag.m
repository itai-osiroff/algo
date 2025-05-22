function [updated_mask, update_step, cost] = perform_g_d_step_disp_diag(LF, disp_mask, Na, x_stp_smp, u_x_max, cost_mask, k_num, k_limit, gd_step, g_filt_var, thresh_cof)

% which step_sizes to check
[ky, kx] = ndgrid(linspace(-k_limit, k_limit, k_num)); % Re-initialize k matrices
ky = gpuArray(ky);
kx = gpuArray(kx);

% initalize gradient and cost matrices
grad_mat = zeros([size(disp_mask, 1), size(disp_mask, 2), k_num, k_num], 'gpuArray');
cost_mat = zeros([k_num, k_num], 'gpuArray');

% Smoothing mask for the gradients
% g_filt_var = 10; % variance of the gaussian filter
g_filt = gpuArray(exp(-0.5/g_filt_var^2*(-round(3*g_filt_var):round(3*g_filt_var)).^2)); % Gaussian filter, zero mean and set variance
g_filt = g_filt./sum(g_filt);

% Iterate through the different gradient step_sizes
aux_ones = ones(size(grad_mat, 1), size(grad_mat, 2), "gpuArray");
aux_ones = conv2(g_filt, g_filt', aux_ones, 'same');
for i = 1:k_num
    for j = 1:k_num
        % if j+(i-1)*k_num == 22
        %     keyboard
        % end
        [grad_mat(:, :, i, j), cost_mat(i, j), ~, ~] = calc_grad_disp_diag(LF, disp_mask, x_stp_smp, u_x_max, cost_mask, [ky(i, j), kx(i, j)], false);
        grad_mat(:, :, i, j) = conv2(g_filt, g_filt', grad_mat(:, :, i, j), 'same')./max(10^-3, aux_ones); % Smooth each gradient per axis per k value
    end
end

% Per k value per pixel, take the value of the gradient corresponding to
% the highest cost
updated_mask = disp_mask;
[~, init_cost, ~, ~] = calc_grad_disp_diag(LF, updated_mask, x_stp_smp, u_x_max, cost_mask, zeros(2, "gpuArray"), false);
[max_vals, max_inds] = max(grad_mat, [], [3, 4], "linear");
[~, ~, max_inds_1, max_inds_2] = ind2sub(size(grad_mat), max_inds);
max_inds = sub2ind(size(kx), max_inds_1, max_inds_2);
% thresh_cof = 0.2;
thresh = max_vals > thresh_cof*mean(grad_mat.*(grad_mat>0), "all"); %Zero out the change for pixels which do not change the cost
% if sum(thresh, "all") == 0
%     keyboard
% end
update_step(:, :, 1) = ky(max_inds).*thresh;
update_step(:, :, 2) = kx(max_inds).*thresh;
% aux_ones = ones(size(update_step), "gpuArray");
aux_ones = thresh;
aux_ones = convn(convn(aux_ones, g_filt, 'same'), g_filt', "same");
%aux_ones_inds = find(aux_ones); %avoid dividing by zero
update_step = convn(convn(update_step, g_filt, 'same'), g_filt', "same");
update_step = update_step./max(10^-3, aux_ones);

% Check smaller gd step_sizes to maximize score
for i = 1:4
    updated_mask_try = updated_mask + gd_step*update_step;
    [~, cost_try, ~, ~] = calc_grad_disp_diag(LF, updated_mask_try, x_stp_smp, u_x_max, cost_mask, zeros(2, "gpuArray"), false);
    if cost_try < init_cost
        gd_step = gd_step / 2;
    else
        updated_mask = updated_mask_try;
        break
    end
end

%return the cost of the final updated_mask
[~, cost, ~, ~] = calc_grad_disp_diag(LF, updated_mask, x_stp_smp, u_x_max, cost_mask, zeros(2, "gpuArray"), false);
% cost = max(init_cost, cost_try);

% if i == 3
%     updated_mask = updated_mask - gd_step*update_step;
% end

% updated_mask = updated_mask - gd_step*update_step;

% Display the corrected LF according to the updated mask
% [~, ~, ~, ~] = calc_grad_disp(LF, Na, updated_mask, x_stp_smp, u_x_max, cost_mask, 0, true);