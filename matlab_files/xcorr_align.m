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
        f1 = 10*10^4; % f of first lens
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
        lf_ang_grid = lf_ang_grid - mean(lf_ang_grid); % Centr the choices
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

%% Loop for each viewpoint and find the best (delta_x, delta_y) which creates the highest correlation
[~, ind] = max(LF, [], "all");
[~, ~, ind1, ind2] = ind2sub(size(LF), ind);
center_LF = squeeze(LF(:, :, ind1, ind2)); % Choose the "central" viewpoint as the one which has the maximally activated pixel
disp_mat = zeros(size(LF, 3), size(LF, 4), 2); %initialize a matrix to hold the displacement
shifted_LF = LF; % Construct a shifted LF according to the deltas found
global_shifted_LF = LF;% Construct a shifted LF according to the median deltas found
for i = 1:size(LF, 3)
    for j = 1:size(LF, 4)
        x_corr = xcorr2_fft(center_LF, squeeze(LF(:, :, i, j)));
        [~, max_ind] = max(x_corr, [], "all");
        [max_sub1, max_sub2] = ind2sub(size(x_corr), max_ind);
        delta_y = max_sub1 - ceil(size(x_corr, 1)/2);
        delta_x = max_sub2 - ceil(size(x_corr, 2)/2);
        disp_mat(i, j, :) = [delta_y, delta_x];
        shifted_LF(:, :, i, j) = circshift(squeeze(LF(:, :, i, j)), [delta_y, delta_x]);
    end
end

%calculate the median deltas and shift the LF
median_delta = floor(squeeze(median(disp_mat, [1, 2])));
% for i = 1:size(LF, 3)
%     for j = 1:size(LF, 4)
%         if i == ind1 && j == ind2
%             continue
%         end
%         global_shifted_LF(:, :, i, j) = circshift(squeeze(global_shifted_LF(:, :, i,j)), median_delta);
%     end
% end

% Display results
displayLF(LF, 1); title("original LF");
displayLF(shifted_LF, 1); title("circshifted LF - Individual corrections");
%displayLF(global_shifted_LF, 1); title("circshifted LF - median (global) corrections");
figure;imagesc(mean(LF, [3,4])); title("LF summation without correction");
figure;imagesc(mean(shifted_LF, [3,4])); title("LF summation with individual correcions");
%figure;imagesc(sum(global_shifted_LF, [3,4])); title("LF summation with median correcions");