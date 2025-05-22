function [init_mask] = initialize_mask(mask_size, mask_type, LF)
% A function which intilizes masks in a more convenient way
if mask_type == "zeros"
    init_mask = zeros(mask_size, "gpuArray");
else
    if mask_type == "random"
        if length(mask_size) == 5 % This is a displacement mask
            init_mask = (rand(mask_size, "gpuArray")-0.5)*10;
        else % This is a slope mask
            init_mask = (rand(mask_size, "gpuArray")-0.5)*0.2;
        end
    end
end

