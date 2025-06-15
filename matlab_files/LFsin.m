function lfT_sin=LFsin(origin_rays, choice, sin_ang_grid, emission_z, lf_ang_grid, x_stp)
lfT_sin = zeros(cat(2, size(origin_rays), 7, 7));
for i = 1:7
    for j = 1:7
        for y = 1:size(origin_rays)[1];
            for x = 1:size(origin_rays)[2];
                if origin_rays(y,x) ~= 1
                    continue;
                end
                % if i == 4 && j==4
                %     print(1)
                % end
                % Original angle is basically determined by which original 
                % ray we're looking at from the original angle grid
                original_angle = [lf_ang_grid(i), lf_ang_grid(j)];
                % We can calculate the spot where this ray hits the
                % aberration layer
                r_o = emission_z/(sqrt(1-original_angle(1)^2-original_angle(2)^2));
                hit_coords = [y, x] + (1/x_stp)*original_angle*r_o;
                if any(hit_coords < 1 | hit_coords > 241)
                    continue
                end

                % The new angle is determined by adding the angle of the
                % sinusoid which was fitted to each pixel in the aberration
                % layer
                y_patch = ceil(hit_coords(1)/15);
                x_patch = ceil(hit_coords(2)/15);
                added_angles = [sin_ang_grid(choice(y_patch, x_patch, 2)), sin_ang_grid(choice(y_patch, x_patch, 1))];
                new_angles = original_angle + added_angles;
                % Calculate the location of the bead according to the new
                % angle using the known z coordinate of the bead plane
                r_n = emission_z/(sqrt(1-new_angles(1)^2-new_angles(2)^2));
                % Multiplication by (1/x_stp) to account for pitch
                %Method 1: Naive approximation
                % new_loc = hit_coords - (1/x_stp)*new_angles*emission_z;
                %Method 2: Using vector analysis
                new_loc = hit_coords - (1/x_stp)*new_angles*r_n;

                % Assign the new location to a pixel in the new image by
                % rounding
                new_loc = round(new_loc);

                %if the calculated location is outside the image then continue
                bad_loc = false;
                for k = 1:2
                    if new_loc(k) < 1 || new_loc(k) > 241
                        bad_loc = true;
                    end
                end
                if bad_loc
                    continue
                end
                % Round angles to fit the original angle grid
                y_ang_diff = added_angles(1);
                x_ang_diff = added_angles(2);
                if 0 <= y_ang_diff
                    if (y_ang_diff < 1/14) || (i == 7)
                        new_angles(y, x, 1) = i;
                    else
                        new_angles(y, x, 1) = i+1;
                    end
                else
                    if y_ang_diff > -1/14 || i == 1
                        new_angles(y, x, 1) = i;
                    else
                        new_angles(y, x, 1) = i-1;
                    end
                end
                if 0 <= x_ang_diff
                    if x_ang_diff < 1/14 || j == 7
                        new_angles(y, x, 2) = j;
                    else
                        new_angles(y, x, 2) = j+1;
                    end
                else
                    if x_ang_diff > -1/14 || j == 1
                        new_angles(y, x, 2) = j;
                    else
                        new_angles(y, x, 2) = j-1;
                    end
                end
                lfT_sin(new_loc(1), new_loc(2), new_angles(y, x, 1), new_angles(y, x, 2)) =+ origin_rays(y, x)*10^(-5);
            end
        end
    end
end