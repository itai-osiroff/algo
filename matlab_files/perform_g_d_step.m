function [updated_mask, update_step, cost] = perform_g_d_step(LF, init_mask, Na, Nxw, smp_stp, stp_ratio, emission_z, x_stp_smp, x_max, lf_ang_grid, u_x_max, cost_mask, k_num, k_limit, gd_step)

% which step_sizes to check
k = gpuArray(linspace(-k_limit, k_limit, k_num)); % k vector

% initalize gradient and cost matrices
grad_mat = zeros([size(init_mask), k_num], 'gpuArray');
cost_vec = zeros([k_num, 1], 'gpuArray');

% Smoothing mask for the gradients
g_filt_var = 10; % variance of the gaussian filter
g_filt = gpuArray(exp(-0.5/g_filt_var^2*(-round(3*g_filt_var):round(3*g_filt_var)).^2)); % Gaussian filter with a range of 11x11 pixels, zero mean and set variance
g_filt = g_filt./sum(g_filt);

% Iterate through the different gradient step_sizes
for i = 1:k_num
    [grad_mat(:, :, :, i), ~, cost_vec(i), ~] = calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, -init_mask, emission_z, x_stp_smp, x_max, lf_ang_grid, u_x_max, cost_mask, k(i), false);
    for ax = 1:2
        grad_mat(:, :, ax, i) = conv2(g_filt, g_filt', grad_mat(:, :, ax, i), 'same'); % Smooth each gradient per axis per k value
    end
end


% Per k value per pixel, take the value of the gradient corresponding to
% the highest cost
updated_mask = init_mask;
[~, max_inds] = max(grad_mat, [], 4);
update_step = k(max_inds);
update_step = convn(convn(update_step, g_filt, 'same'), g_filt', "same");
% for i1 = 1:size(grad_mat, 1)
%     for i2 = 1:size(grad_mat, 2)
%         for ax = 1:size(grad_mat, 3)
%             [~, max_ind] = max(grad_mat(i1, i2, ax);
%             update_step(i1, i2, ax) = grad_mat(i1 ,i2, ax, max_ind);
%         end
%     end
% end


% Check smaller gd step_sizes to maximize score
[~, init_cost, ~, ~] = calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, -init_mask, emission_z, x_stp_smp, x_max, lf_ang_grid, u_x_max, cost_mask, 0, false);
for i = 1:4
    updated_mask_try = updated_mask - gd_step*update_step;
    [~, ~, cost_try, ~] = calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, -updated_mask_try, emission_z, x_stp_smp, x_max, lf_ang_grid, u_x_max, cost_mask, 0, false);
    if cost_try < init_cost
        gd_step = gd_step / 2;
    else
        updated_mask = updated_mask_try;
        break
    end
end

%return the cost of the final updated_mask
[~, ~, cost, ~] = calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, -updated_mask, emission_z, x_stp_smp, x_max, lf_ang_grid, u_x_max, cost_mask, 0, false);
% cost = max(init_cost, cost_try);

% if i == 3
%     updated_mask = updated_mask - gd_step*update_step;
% end

% updated_mask = updated_mask - gd_step*update_step;

% Display the corrected LF according to the updated mask
% [~, ~, ~, ~] = calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, -updated_mask, emission_z, x_stp_smp, x_max, lf_ang_grid, u_x_max, cost_mask, 0, true);