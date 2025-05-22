function all_images=show_LF_grid(LF)

lf_size = size(LF);
size_y = lf_size(1);
size_x = lf_size(2);
all_images = ones((size_y+10)*7, (size_x+10)*7)*2*10^(-5);
image_start_y = 6;
image_start_x = 6;
image_stop_y = size_y+5;
image_stop_x = size_x+5;
for j_v_x = 1:7
    image_start_y = 6;
    image_stop_y = size_y+5;
    for j_v_y = 1:7
        all_images(image_start_y:image_stop_y, image_start_x:image_stop_x) = squeeze(LF(:, :, j_v_y, j_v_x));
        image_start_y = image_start_y + 251;
        image_stop_y = image_stop_y + 251;
    end
    image_start_x = image_start_x + 251;
    image_stop_x = image_stop_x + 251;
end
figure;
imagesc(real(all_images))
x_ticks = round((6+246)/2):251:1757;
y_ticks = round((6+246)/2):251:1757;
x_ticks_l = string(1:7);
y_ticks_l = string(1:7);
xticks(x_ticks)
yticks(y_ticks)
xticklabels(x_ticks_l)
yticklabels(y_ticks_l)
xlabel("v_x")
ylabel("v_y")
title("Light Field Grid - all viewpoints")