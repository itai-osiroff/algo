function [init_mask, gd_cost, opt_over_early] = perform_gd(LF, init_mask, mask_s, gd_iter_num, Na, Nxw, smp_stp, stp_ratio, z_abrr, x_stp, x_max, lf_ang_grid, u_x_max, cost_mask, k_num, k_limit, gd_step)

gd_cost = zeros([gd_iter_num, 1], "gpuArray");
grads = zeros([gd_iter_num, size(mask_s)]);
opt_over_early = 0; % Flag which indicates if the optimization was over before reaching gd_iter_num
for i = 1:gd_iter_num
    [init_mask_new, grads(i, :, :, :), gd_cost(i)] = perform_g_d_step(LF, init_mask, Na, Nxw, smp_stp, stp_ratio, z_abrr, x_stp*smp_stp, x_max, lf_ang_grid, u_x_max, cost_mask, k_num, k_limit, gd_step);
    [~, ~, gd_cost(i), ~] = calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, -init_mask, z_abrr, x_stp*smp_stp, x_max, lf_ang_grid, u_x_max, cost_mask,0, false);

    % If the new mask after GD step is the same as the mask before the step
    % - optimization is over
    if init_mask_new == init_mask
        opt_over_early = i; % flag takes the value of the last iteration done
        break
    else
        init_mask = init_mask_new;
    end
    
    % Show masks in both directions every other iteration
    if mod(i, 2) == 0
        % figure;
        % subplot(1,2,1); imagesc(init_mask(:, :, 1)); title("GD mask x-axis, iter = " + i);
        % subplot(1,2,2); imagesc(init_mask(:, :, 2)); title("GD mask y-axis, iter = " + i);
        % [~, ~, ~, ~] = calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, -init_mask, z_abrr, x_stp*smp_stp, x_max, lf_ang_grid, u_x_max, cost_mask, 0, true);
    end
end