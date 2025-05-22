delta = 5;
flag = 0;
for i1 = 1:Na
    for i2 = 1:Na
        for i = [20, 40]
            for ax = [1, 2]
                init_mask_1 = zeros(size(init_mask), "gpuArray");
                init_mask_1 = 0.5*gt_disp;
                [grad, ~, ~, ~] = calc_grad_disp(LF, Na, init_mask_1, x_stp*smp_stp, u_x_max, cost_mask, delta, false);
                inds = find(grad(:, :, i1, i2, ax));
                if not(any(inds))
                    continue
                end
                ind = inds(randi(end));
                [j, k] = ind2sub(size(squeeze(grad(:, :, i1, i2, ax))), ind);
                [~, cost_pre, ~, ~] = calc_grad_disp(LF, Na, init_mask_1, x_stp*smp_stp, u_x_max, cost_mask, 0, false);
                init_mask_1(j, k, i1, i2, ax) = init_mask_1(j, k, i1, i2, ax) + delta;
                [~, cost_post, ~, ~] = calc_grad_disp(LF, Na, init_mask_1, x_stp*smp_stp, u_x_max, cost_mask, 0, false);
                fprintf("Numerical gradient for (%.0f, %.0f) pixel in viewpoint (%.0f, %.0f) ax=%.0f is: %e\n", j, k, i1, i2, ax, (cost_post-cost_pre))
                fprintf("The claculated gradient by the function is: %e\n", grad(j, k, i1, i2, ax))
                if abs(grad(j, k, i1, i2, ax) - (cost_post-cost_pre)) > min(abs(grad(j, k, i1, i2, ax)), abs((cost_post-cost_pre)))*0.00001
                    fprintf("Gradients don't match\n")
                    flag = 1;
                end
            end
        end
    end
end

if flag
    fprintf("Gradients weren't identical")
else
    fprintf("Gradients were identical")
end