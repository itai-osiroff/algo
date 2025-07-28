% Load an existing result of the simulator or calculate a new one
if input("Use pre-computed random abberation layer? (y/n)", "s") == "y"
    load example_LF.mat
else
    if input("Use pre-computed sinus abberation layer? (y/n)", "s") == "y"
        load LF_sinus.mat
    else
        if input("Use pre-computed parabolic abberation layer? (y/n)", "s") == "y"
            load LF_parabolic.mat
        else
            test_BuildTMat_LF_smp;
        end
    end
end
% Making sure the phase mask matrix is gpuArray
if class(mask_s) ~= "gpuArray"
    mask_s = gpuArray(mask_s);
end

ind=(ceil(Nl/2)); % index of the central bead
step_size = 5*10^(-2); % step size for the gradient

% cost mask initilization and 1/9 bead illumination
cost_mask = zeros(length(LF1), length(LF1)); %initilization of the cost mask (where are the points)
if input("Use 1 bead illumination or 9 point illumination? (1/9)", "s") == "1"
    cost_mask(ceil(length(LF1)/2), ceil(length(LF1)/2)) = 1; % cost mask measures at the center pixel
else
    [row, col] = find(reshape(sum(uin,2), sqrt(length(uin)), sqrt(length(uin))));
    for i = row(:)
        for j = col(:)
            cost_mask(ceil(i/smp_stp), ceil(j/smp_stp)) = 1; % cost mask measures at the center pixel
        end
    end
end
cost_mask = gpuArray(cost_mask); % Some of the indices in [row,col] repeat so cost_mask is initialized on the CPU then transferred to GPU

% Smoothing the cost mask
g_filt = gpuArray(exp(-0.5/1.5^2*(-5:5).^2)); %1.5 is the variance of the gaussian
cost_mask = conv2(g_filt, g_filt', cost_mask, "same");

% Calculating the numerical gradient per pixel per axis (in the first
% comment, there is the initial function used) - sparse sampling

%num_grad = calc_num_grad(LF1, Na, Nxw, smp_stp, -mask_s, z_abrr, x_stp*smp_stp, x_max, lf_ang_grid, u_x_max, cost_mask, step_size);
diff_mask_1 = zeros(size(mask_s), "gpuArray");
diff_mask_2 = zeros(size(mask_s), "gpuArray");
num_grad = zeros(size(mask_s), "gpuArray");
for i1 = 1:20:(Nxw+smp_stp-1)/smp_stp
    for i2 = 1:20:(Nxw+smp_stp-1)/smp_stp
        for ax = 1:2
            diff_mask_1(i1, i2, ax) = 1;
            diff_mask_2(i1, i2, ax) = 1;
            mask_s_t = -mask_s;
            mask_s_t(i1, i2, ax) = mask_s_t(i1, i2, ax) + step_size;
            [~, cost_0] = calc_grad(LF1, Na, Nxw, smp_stp, -mask_s, z_abrr, x_stp*smp_stp, x_max, lf_ang_grid, u_x_max, cost_mask, step_size);
            [~, cost_1] = calc_grad(LF1, Na, Nxw, smp_stp, mask_s_t, z_abrr, x_stp*smp_stp, x_max, lf_ang_grid, u_x_max, cost_mask, step_size);
            num_grad(i1, i2, ax) = (cost_1 - cost_0)/step_size;
        end
    end
end

%Calculating the fast gradient
grad = calc_grad(LF1, Na, Nxw, smp_stp, -mask_s, z_abrr, x_stp*smp_stp, x_max, lf_ang_grid, u_x_max, cost_mask, step_size);

% Showing the differences between fast and slow gradient where it was
% sampled

% diff_mask_1 = im2bw(abs(num_grad(:, :, 1)), 10^-10);
% diff_mask_2 = im2bw(abs(num_grad(:, :, 2)), 10^-10);
diff_map_1 = num_grad(:, :, 1) - grad(:, :, 1).*diff_mask_1(:, :, 1);
diff_map_2 = num_grad(:, :, 2) - grad(:, :, 2).*diff_mask_2(:, :, 2);
figure; imagesc(grad(:, :, 1)); title("Fast gradient, axis=1")
figure; imagesc(grad(:, :, 2)); title("Fast gradient, axis=2")
figure; imagesc(diff_map_1); title("difference map, axis=1")
figure; imagesc(diff_map_2); title("difference map, axis=2")
plot(num_grad(find(diff_mask_1)))
hold on
plot(grad(find(diff_mask_1)))