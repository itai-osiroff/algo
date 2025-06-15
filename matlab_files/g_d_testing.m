% Load an existing result of the simulator or calculate a new one
if input("Use light field Marina got? (y/n)", "s") == "y"
    load 2024_03_05_10_18_10_chicken_light_field_BSI_4beads.mat
else
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
end

% Is this used for testing from simulator
if exist("mask_s", "var")
    % Making sure the phase mask matrix is gpuArray
    if class(mask_s) ~= "gpuArray"
        mask_s = gpuArray(mask_s);
    end

    stp_ratio = 1; % ratio between the step_sizes of u_x_max and x_max
    ind=(ceil(Nl/2)); % index of the central bead
    
    % cost mask initilization and 1/9 bead illumination
    cost_mask = zeros(length(LF1), length(LF1)); %initilization of the cost mask (where are the points)
    if input("Use 1 bead illumination or 9 point illumination? (1/9)", "s") == "1"
        cost_mask(ceil(length(LF1)/2), ceil(length(LF1)/2)) = 1; % cost mask measures at the center pixel
        LF = LF1;
    else
        [row, col] = find(reshape(sum(uin,2), sqrt(length(uin)), sqrt(length(uin))));
        for i = row(:)
            for j = col(:)
                cost_mask(ceil(i/smp_stp), ceil(j/smp_stp)) = 1; % cost mask measures at the center pixel
            end
        end
        LF = LF9;
    end
    cost_mask = gpuArray(cost_mask); % Some of the indices in [row,col] repeat so cost_mask is initialized on the CPU then transferred to GPU
    
    % Smoothing the cost mask
    g_filt = gpuArray(exp(-0.5/1.5^2*(-5:5).^2)); %1.5 is the variance of the gaussian
    cost_mask = conv2(g_filt, g_filt', cost_mask, "same");
else
    if input("Use precalculated LF and cost_mask? (y/n)", "s") == "n"
        f1 = 12.5*10^4; % f of first lens
        f2 = 20*10^4; % f of second lens
        f3 = 1*10^4; % f of thrid lens (objective)
        slm_pix_size = 8; % Size of each pixel in the SLM
        pos_ctrs = APT.apt_pos_ctrs(1:5, 1);% Relevant coordinates of the centers of the light-field apertures on the SLM (only first column of windows)
        % pos_ctrs = pos_ctrs - pos_ctrs(ceil(length(pos_ctrs)/2)); % Transforming to displacement from center
        % lf_ang_grid = (pos_ctrs*slm_pix_size)/(f1) * 20; % Transforming displacements to sin(a) (angles), 20 is the magnification
        left_pix = 92; % This pixel represents in Fourier space sin(theta)=-0.5, and is the left-most pixel to enter the NA
        right_pix = 310; % This pixel represents in Fourier space sin(theta)=0.5, and is the right-most pixel to enter the NA
        ang_res = 9; %resolution of the angle search within [left_pix, right_pix]
        lf_ang_grid = -0.5:1/ang_res:0.5; % initialize the angle grid for search
        lf_ang_grid=lf_ang_grid-mean(lf_ang_grid);
        bin_size = (right_pix-left_pix)/ang_res; % divide the search space to bins acoording to the resolution
        bin_locs = ceil((pos_ctrs-left_pix)/bin_size); % allocate each center_pos to a bin
        lf_ang_grid = lf_ang_grid(bin_locs); % Decide which angle goes to each pos_ctr
        lf_ang_grid = lf_ang_grid - mean(lf_ang_grid); % Center the choices
        Na = sqrt(size(Images_apt, 3));
        lambda = 0.52;
        x_stp = 0.27;
        LF = zeros([size(Images_apt, 1), size(Images_apt, 2), sqrt(size(Images_apt, 3)), sqrt(size(Images_apt, 3))], "gpuArray");
        for i = 1:sqrt(size(Images_apt, 3))
            for j = 1:sqrt(size(Images_apt, 3))
                LF(:, :, j, i) = Images_apt(:, :, (i-1)*sqrt(size(Images_apt, 3))+j);
            end
        end
    
        % Resize validation camera image to fit the sampling camera size\
        I_val = imresize(I_val, 0.6758, "bilinear");
        I_val = max(I_val - median(I_val(:))*4, 0);
    
        % Find the shift (in pixels) which gives highest correlation between
        % the center light-field and the validation
        center_LF = Images_apt(:, :, 13);
        n1 = size(center_LF, 1);
        n2 = size(center_LF, 2);
        offset = (1/3)*length(I_val); % We don't need to measure correlation with the entire cost_mask, only with about 2/3 of it
        I_val_corr = zeros([size(I_val, 1)+n1*2, size(I_val, 2)+n2*2], "gpuArray");
        I_val_corr(n1+1:n1+size(I_val, 1),n2+1:n2+size(I_val, 2)) = I_val(:,:);
        best_corr = 0; %initialize best correlation holder
        best_corr_inds = [0, 0]; %initialize best correlation indices holder
        for i = 1:size(I_val, 1)+n1
            for j = 1:size(I_val, 2)+n2
                corr = sum(sum(I_val_corr(i:i+n1-1, j:j+n2-1).*center_LF)); % calculate correlation
                corr_m(i, j) = corr;
                if corr > best_corr
                    best_corr_inds = [i, j];
                    best_corr = corr;
                end
            end
        end
    
        I_val_cut = I_val_corr(best_corr_inds(1)+1:best_corr_inds(1)+n1, best_corr_inds(2)+1: best_corr_inds(2)+n2);
    
        % cost_mask = zeros(length(LF), length(LF)); %initilization of the cost mask (where are the points)
        cost_mask = gpuArray(I_val_cut); % Cost mask is the validation image
        % cost_mask = gpuArray(cost_mask); % Legacy: Some of the indices in [row,col] repeat so cost_mask is initialized on the CPU then transferred to GPU
        
        % Smoothing the cost mask
        g_filt = gpuArray(exp(-0.5/2^2*(-8:8).^2)); 
        cost_mask = conv2(g_filt, g_filt', cost_mask, "same");
        cost_mask = max(cost_mask - median(cost_mask(:))*2, 0);
    else
        load marina_4_beads_ws.mat
    end
    u_x_max = floor(length(cost_mask)/2)*x_stp; % = floor(num of pixels * x_stp/2) num of pixels should be odd
    
    % Try GD with different z_abrr values
    z_abrr = linspace(30, 500, 20);

    x_max = u_x_max + ceil(0.5*max(z_abrr)/x_stp)*x_stp; % add the numerical aperture dilation to the aberration layer
    Nxw = x_max/x_stp*2+1;
    stp_ratio = 1; % ratio between the step_sizes of u_x_max and x_max
    smp_stp = 1;
end
% Initialize a slightly perturbated phase mask as a starting point for the
% G.D, by adding randomly generated perturbations on the order of 0.5% of the
% highest absolute value of the real mask
if exist("mask_s", "var")
    init_mask_per = mask_s + max(abs(mask_s), [], "all")*0.005*rand(size(mask_s), "gpuArray");
    init_mask = zeros(size(mask_s), "gpuArray");
else
    init_mask = zeros([size(I_val_cut), 2], "gpuArray");
    init_size = x_max/(x_stp*stp_ratio)*2+1;
    init_mask = zeros(init_size,init_size,2, "gpuArray");
end

% Perform GD iterations and plot summed difference from the real mask
gd_iter_num = 20; % number of GD steps
k_limit = 0.1; % Range of gradient step_size values
k_num = 8; % resolution of sampling in the range
gd_step = 1; % G.D step size

% for k_num = 6:2:16
%     for k_limit = linspace(0.01, 0.2, 15)
%         if k_num < 10
%             continue 
%         end
%         init_mask = zeros(init_size,init_size,2, "gpuArray");
%         gd_cost = zeros([gd_iter_num, 1], "gpuArray");
%         [init_mask, gd_cost, opt_over_early] = perform_gd(LF, init_mask, zeros(size(init_mask), "gpuArray"), gd_iter_num, Na, Nxw, smp_stp, stp_ratio, z_abrr(4), x_stp, x_max, lf_ang_grid, u_x_max, cost_mask, k_num, k_limit, gd_step);
%         [~, ~, ~, ~] = calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, -init_mask, z_abrr(4), x_stp*smp_stp, x_max, lf_ang_grid, u_x_max, cost_mask, 0, true);
%         figure; imagesc(squeeze(init_mask(:, :, 1))); title("mask ax=1 for k_num=" + k_num + " k_limit=" + k_limit);
%         figure; imagesc(squeeze(init_mask(:, :, 2))); title("mask ax=2 for k_num=" + k_num + " k_limit=" + k_limit);
%         figure; hold on; plot(1:gd_iter_num, gd_cost, "r-o", "DisplayName", "w.r.t initial score");
%         [maximum, maxloc] = max(gd_cost);
%         xline(maxloc, "Color", "r", "LineWidth", 2)
%         textLabel = sprintf("Max at iter=%.0f of %.5e", maxloc, maximum);
%         text(3, gd_cost(3), textLabel, "FontSize", 15, "Color", "r")
%         xlabel("GD iteration #"); ylabel("score");
%         title("k_limit = " + k_limit + " k_num = " + k_num)
%         legend show
%         hold off
%         % save figues
%         folderPath = "/home/itaio/new_sim_backward/images/z_abrr_testing/optim_z_100";
%         resolution = 300;
% 
%         if ~exist(folderPath, 'dir')
%             mkdir(folderPath);
%         end
% 
%         figHandles = findall(groot, 'Type', 'figure');
% 
%         for i=1:length(figHandles)
%             figHandle = figHandles(i);
%             fileName = fullfile(folderPath, ['k_num=' num2str(k_num) '_k_limit=' num2str(k_limit) '_Figure_' num2str(figHandle.Number) '.png']);
%             exportgraphics(figHandle, fileName, 'Resolution', resolution)
%         end
%         close all
%     end
% end
% keyboard

if size(z_abrr) == 1
    [~, ~, ~, ~] = calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, -init_mask, z_abrr, x_stp*smp_stp, x_max, lf_ang_grid, u_x_max, cost_mask, 0, true);
    [init_mask, gd_cost, opt_over_early] = perform_gd(LF, init_mask, mask_s, gd_iter_num, Na, Nxw, smp_stp, stp_ratio, z_abrr, x_stp, x_max, lf_ang_grid, u_x_max, cost_mask, k_num, k_limit, gd_step);
    [~, ~, ~, ~] = calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, -init_mask, z_abrr, x_stp*smp_stp, x_max, lf_ang_grid, u_x_max, cost_mask, 0, true);
else
    masks = zeros([length(z_abrr), size(init_mask)], "gpuArray");
    for i = 1:length(z_abrr)
        % if not(i==5 || i==10 || i==15 || i==20)
        %     continue
        % end
        % displayLF(LF, 1);
        % LF = LF.*(LF>2*median(LF, "all"));
        displayLF(LF, 1);
        init_mask = zeros(init_size,init_size,2, "gpuArray");
        [init_mask, gd_cost, opt_over_early] = perform_gd(LF, init_mask, zeros(size(init_mask), "gpuArray"), gd_iter_num, Na, Nxw, smp_stp, stp_ratio, z_abrr(i), x_stp, x_max, lf_ang_grid, u_x_max, cost_mask, k_num, k_limit, gd_step);
        masks(i, :, :, :) = init_mask;
        figure; imagesc(squeeze(masks(i, :, :, 1))); title("mask ax=1 for z=" + z_abrr(i));
        figure; imagesc(squeeze(masks(i, :, :, 2))); title("mask ax=2 for z=" + z_abrr(i));
        [~, ~, ~, ~] = calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, -init_mask, z_abrr(i), x_stp*smp_stp, x_max, lf_ang_grid, u_x_max, cost_mask, 0, true);
        figure; hold on; plot(1:gd_iter_num, gd_cost, "r-o");
        keyboard
    end
end

% for i = 1:size(masks, 1)
%     figure; imagesc(squeeze(masks(i, :, :, 1))); title("mask ax=1 for z=" + z_abrr(i));
%     figure; imagesc(squeeze(masks(i, :, :, 2))); title("mask ax=2 for z=" + z_abrr(i));
% end

keyboard

if exist("mask_s", "var")
    [~, ideal_cost, ~, ~] = calc_grad(LF, Na, Nxw, smp_stp, stp_ratio, -mask_s, z_abrr, x_stp*smp_stp, x_max, lf_ang_grid, u_x_max, cost_mask, 0, false); % The so-called ideal cost, score of the mask from the simulation
end
init_cost = min(gd_cost, [], "all"); % initial cost of init_mask
figure; hold on; plot(1:gd_iter_num, (gd_cost-ones(size(gd_cost))*init_cost)/init_cost*100, "r-o", "DisplayName", "w.r.t initial score");
figure; hold on; plot(1:gd_iter_num, gd_cost, "r-o", "DisplayName", "w.r.t initial score");
% plot(1:gd_iter_num, gd_cost/ideal_cost*100, "b-o", "DisplayName", "w.r.t score of the simulation mask")
xlabel("GD iteration #"); ylabel("% improvement");
legend show
hold off