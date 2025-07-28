function [init_mask, gd_cost, opt_over_early] = perform_gd_disp_diag(LF, init_mask, gd_iter_num, Na, smp_stp, x_stp, u_x_max, cost_mask, k_num, k_limit, gd_step, g_filt_var, thresh_cof)

gd_cost = zeros([gd_iter_num, size(LF, 3), size(LF, 4)], "gpuArray");
% opt_over_early = 0; % Flag which indicates if the optimization was over before reaching gd_iter_num

for i1 = 1:size(LF, 3)
    for i2 = 1:size(LF, 4)
        if i1 == 3 && i2 == 2
            keyboard
        end
        grads = zeros([gd_iter_num, size(init_mask, 1), size(init_mask, 2), 2], "gpuArray"); %Re-initialiae the gradients for each viewpoint
        % init_mask_new = zeros([size(init_mask, 1), size(init_mask, 2), 2], "gpuArray"); % Re-initialize disp mask for each viewpoint
        k_limit_iter = k_limit; %initialize a k vector for each viewpoint separately
        for i = 1:gd_iter_num
            
            [init_mask_new, grads(i, :, :, :), gd_cost(i, i1, i2)] = perform_g_d_step_disp_diag(squeeze(LF(:, :, i1, i2)), squeeze(init_mask(:, :, i1, i2, :)), Na, x_stp*smp_stp, u_x_max, cost_mask, k_num, k_limit_iter, gd_step, g_filt_var, thresh_cof);
            [~, gd_cost(i, i1, i2), ~, ~] = calc_grad_disp_diag(squeeze(LF(:, :, i1, i2)), init_mask_new, x_stp*smp_stp, u_x_max, cost_mask, zeros(2, "gpuArray"), false);
            
            % If the new mask after GD step is the same as the mask before the step
            % - optimization is over
            if init_mask_new == squeeze(init_mask(:, :, i1, i2, :))
                opt_over_early = i; % flag takes the value of the last iteration done
                break
            else
                init_mask(:, :, i1, i2, :) = init_mask_new;
            end
            
            k_limit_iter = k_limit_iter*0.9; % k_limit scheduling

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

    end
end
