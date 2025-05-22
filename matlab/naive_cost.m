function cost=naive_cost(LF, Na, mask_s, emission_z, x_stp_smp, x_max, lf_ang_grid, u_x_max, cost_mask)

% Based on a given LF and slopes mask, we calculate the original LF
origin_LF = calc_back(LF, Na, -mask_s, emission_z, x_stp_smp, x_max, lf_ang_grid, u_x_max);

% Calculate the illumination pattern
illum_pattern_calc = mean(mean(origin_LF,3), 4);

% Calculate the cost by summing the energy fraction in the indices of the
% illuminated pixels (matrix of 9 points)
cost = sum(sum(illum_pattern_calc.*cost_mask));