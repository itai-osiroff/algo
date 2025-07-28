function [x, cost_history] = adam_optim(LF, Na, x_init, x_stp, smp_stp, u_x_max, cost_mask, learning_rate, beta1, beta2, epsilon, max_iter)
 %initialize the parameters
 x = x_init;
 m = zeros(size(x), "gpuArray");
 v = zeros(size(x), "gpuArray");
 t = 0;

cost_history = zeros(max_iter,1, "gpuArray");

for i = 1:max_iter
    t = t+1;

    % Computer gradient
    [grad, ~, ~, ~] = calc_grad_disp(LF, Na, x, x_stp*smp_stp, u_x_max, cost_mask,0, false);
    displayLF(squeeze(grad(:, :, :, :, 1)), 1);
    displayLF(squeeze(grad(:, :, :, :, 2)), 1);
    %update biased first moment estimate
    m = beta1*m + (1-beta1)*grad;

    %computer biased second moment estimate
    v = beta2*v + (1-beta2)* (grad.^2);

    %computer bias-corrected first moment estimate
    m_hat = m/ (1-beta1^t);

    %compute bias-corrected second moment estimate
    v_hat = v/ (1-beta2^t);
    
    % update parameters 
    x = x + learning_rate * m_hat ./ (sqrt(v_hat)+epsilon);

    %calculate the currect cost
    [~, ~, cost_history(i), ~] = calc_grad_disp(LF, Na, x, x_stp*smp_stp, u_x_max, cost_mask,0, false);

    % display progress
    fprintf("Iteration %d, cost: %.6f" + newline, i, cost_history(i));

end
end