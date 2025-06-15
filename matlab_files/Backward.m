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

if input("Use 1 bead illumination or 9 point illumination? (1/9)", "s") == "1"
    ind=(ceil(Nl/2)); % index of the central bead
    
    %Na = number of viewpoints, Nx_smp = resolution of the LF, mask_s = slope
    %matrix fitted to the aberration layer
    
    % Display a grid of 2D images corresponding to the light field grid of
    % viewpoints
    % Concatenate the images on a grid
    displayLF(LF1,1); title('light field 1 bead')
    
    % Calculate the original illumination positions
    % Note: we input in the function calc_back (-mask_s) because we want to
    % correct the phase shift from the aberration layer
    origin_LF = calc_back(LF1, Na, -mask_s, z_abrr, x_stp*smp_stp, x_max, lf_ang_grid, u_x_max);
    displayLF(origin_LF, 1); title('light field corrected');
    illum_pattern_calc = mean(mean(origin_LF,3),4);
    
    % % Display real illumination pattern, calculated illumination pattern, and
    % % the difference
    illum_pattern_real = reshape(uin(:, ind), sqrt(length(uin)), sqrt(length(uin)));
else % The same but for 9 point illumination
    displayLF(LF9,1); title('light field 9 beads')
    origin_LF = calc_back(LF9, Na, -mask_s, z_abrr, x_stp*smp_stp, x_max, lf_ang_grid, u_x_max);
    displayLF(origin_LF, 1); title('light field corrected')
    illum_pattern_calc = mean(mean(origin_LF,3), 4);
    illum_pattern_real = reshape(sum(uin,2), sqrt(length(uin)), sqrt(length(uin)));
end
figure; imagesc(illum_pattern_real); title("Real Illumination");
figure; imagesc(illum_pattern_calc); title("Calculated illumination");
displayLF_real(LF1-origin_LF, 1); title("Difference between calculated and real illumination patterns");