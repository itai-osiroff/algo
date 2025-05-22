function [init_mask, gd_cost, opt_over_early] = perform_gd_disp(LF, init_mask, gd_iter_num, Na, smp_stp, x_stp, u_x_max, cost_mask, k_num, k_limit, gd_step, g_filt_var, thresh_cof)

gd_cost = zeros([gd_iter_num, 1], "gpuArray");
grads = zeros([gd_iter_num, size(init_mask)]);
opt_over_early = 0; % Flag which indicates if the optimization was over before reaching gd_iter_num
for i = 1:gd_iter_num
    [init_mask_new, grads(i, :, :, :, :, :), gd_cost(i)] = perform_g_d_step_disp(LF, init_mask, Na, x_stp*smp_stp, u_x_max, cost_mask, k_num, k_limit, gd_step, g_filt_var, thresh_cof);
    [~, gd_cost(i), ~, ~] = calc_grad_disp(LF, Na, init_mask_new, x_stp*smp_stp, u_x_max, cost_mask,0, false);

    
    % If the new mask after GD step is the same as the mask before the step
    % - optimization is over
    if init_mask_new == init_mask
        opt_over_early = i; % flag takes the value of the last iteration done
        break
    else
        init_mask = init_mask_new;
    end
    
    k_limit = k_limit*0.9;
    
    % Add perturbations every 5th iteratio to try and get out of local
    % minima
    % if mod(i, 5) == 0
    %     change = (rand(size(init_mask), "gpuArray")-0.5)*0.2*max(abs(init_mask), [], "all");
    %     new_init_mask = init_mask + change;
    %     init_mask = new_init_mask;
    % end

    %Show masks in both directions every other iteration
    % if mod(i, 2) == 0
    %     displayLF(squeeze(init_mask(:, :, :, :, 1)), 1);
    %     title("displacement masks ax=1")
    %     displayLF(squeeze(init_mask(:, :, :, :, 2)), 1);
    %     title("displacement masks ax=2")
    %     [~, ~, ~, ~] = calc_grad_disp(LF, Na, init_mask, x_stp*smp_stp, u_x_max, cost_mask, 0, true);
    % end
end