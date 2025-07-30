import os
import PIL
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from matplotlib import cm

def load_image(image_path):
    """
    Load an image from the specified path.
    
    :param image_path: Path to the image file.
    :return: PIL Image object or None if the image cannot be loaded.
    """
    if not os.path.exists(image_path):
        print(f"Error: The file {image_path} does not exist.")
        return None
    
    try:
        image = Image.open(image_path)
        return image
    except PIL.UnidentifiedImageError:
        print(f"Error: The file {image_path} is not a valid image.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the image: {e}")
        return None


def min_max_normalize(image, min=0, max=255):
    """
    Normalize the image using min-max normalization. 
    
    :param image: PIL Image object.
    :return: Normalized PIL Image object.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object.")
    
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Perform min-max normalization
    min_val = image_array.min()
    if min > 0:
        min_val = min
    else:
        min_val = image_array.min()
    max_val = image_array.max()
    if max < 255:
        max_val = max
    else:
        max_val = image_array.max()
    
    if min_val == max_val:
        print("Warning: The image has no variation in pixel values. Returning a blank image.")
        return image.copy()
    else:
        normalized_array = (image_array - min_val) / (max_val - min_val) * 255
        normalized_image = Image.fromarray(normalized_array.astype('uint8'))
    
    return normalized_image

if __name__ == "__main__":
    images=[]
    ims_dir = 'moment_images/images'
    for f in os.listdir(ims_dir):
        image_path = os.path.join(ims_dir, f)
        image = load_image(image_path)
        if image:
            images.append(image)
            print(f"Successfully loaded image: {f}")
        else:
            print(f"Failed to load image: {f}")
    for i, im in enumerate(images):
        norm_im = min_max_normalize(im).rotate(-5)
        # norm_im = min_max_normalize(im)
        images[i] = (im, norm_im)
        # norm_im.show()
    circles_chosen = None
    search_rad_min = 50
    search_rad_max = 120
    for i, im in enumerate(images):
        cimg = cv.cvtColor(np.array(im[1]),cv.COLOR_GRAY2BGR)
        im_np = np.array(im[1])
        roi_choice = input("Enter ROI strategy (1: All of the image, 2: Center of the image, 3: Centered on maximal pixel): ")
        if roi_choice == "2":
            # ROI = Center of the image
            roi = im_np[np.shape(im_np)[0]//2-search_rad_max*2:np.shape(im_np)[0]//2+search_rad_max*2, np.shape(im_np)[1]//2-search_rad_max*2:np.shape(im_np)[1]//2+search_rad_max*2]
        elif roi_choice == "3":
            # ROI = centered on the maximal pixel
            max_search_im = np.array(cv.GaussianBlur(np.array(im[0]), (51, 51), sigmaX=5, sigmaY=5))
            max_loc = np.unravel_index(np.argmax(max_search_im), np.shape(max_search_im))
            roi = im_np[max_loc[0]-search_rad_max*2:max_loc[0]+search_rad_max*2, max_loc[1]-search_rad_max*2:max_loc[1]+search_rad_max*2]
        else:
            # ROI = all of image
            roi = im_np
        circles = cv.HoughCircles(roi, cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=search_rad_min, maxRadius=search_rad_max)
        if circles is None or len(circles) == 0:
            print(f"No circles found in image {i+1}.")
            continue
        circles = np.around(circles).astype(np.int64)
        if roi_choice == "2":
            for i in range(len(circles[0])):
                circles[0, i, 0] += (np.shape(im_np)[0]//2 - search_rad_max*2)
                circles[0, i, 1] += (np.shape(im_np)[1]//2 - search_rad_max*2)
        elif roi_choice == "3":
            for i in range(len(circles[0])):
                circles[0, i, 0] += (max_loc[0] - search_rad_max*2)
                circles[0, i, 1] += (max_loc[1] - search_rad_max*2)
        for i in circles[0,:2]: # only draw most voted 2 circles
            # draw the outer circle
            cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        
        # cv.namedWindow(f'image {i+1}', cv.WINDOW_NORMAL)
        # cv.imshow(f'image {i+1}', cimg)
        # cv.waitKey(0)
        plt.figure()
        plt.imshow(cimg)
        plt.show()
        if input("Use this image for further processing? (y/n): ").lower() =="y":
            circles_chosen = circles
            norm_im = im_np
            break
        plt.close("all")
    # cv.destroyAllWindows()
    if circles_chosen is None:
        circles_chosen = circles

    # Let's use the first 2 circles for one of the images found as the basis for the other circles assuming a square grid
    windows = circles_chosen[0, :2]
    # d = int(np.linalg.norm(windows[0][:2] - windows[1][:2])/np.sqrt(2)) # Use 2 of the first detected circles
    d = np.int_(np.floor(1.1*2*np.max(windows[:, 2]))) # Use the maximum radius of the circles found
    sub_im_num = 3
    views = np.zeros((sub_im_num, sub_im_num, d, d))
    for i in range(0, sub_im_num):
        for j in range(0, sub_im_num):
            x = int(windows[0][0] + (i-sub_im_num//2) * d)
            y = int(windows[0][1] + (j-sub_im_num//2) * d)
            if x - (d//2) < 0 or y - (d//2) < 0 or x + (d//2) >= np.shape(norm_im)[0] or y + (d//2) >= np.shape(norm_im)[1]:
                print(f"Skipping view {i+1}-{j+1} due to out of bounds coordinates.")
                continue
            if (d//2)*2 == np.shape(views)[2]:
                views[i, j, :, :] = np.array(norm_im)[y-(d//2):y+(d//2), x-d//2:x+d//2]
            else:
                views[i, j, :, :] = np.array(norm_im)[y-(d//2):y+(d//2)+1, x-d//2:x+d//2+1]

    for i in range(0, sub_im_num):
        for j in range(0, sub_im_num):    
            view_for_imshow = np.array(min_max_normalize(Image.fromarray(views[i, j])))
            plt.figure()
            plt.imshow(view_for_imshow, cmap="gray")
    plt.show()
    plt.close("all")
    # Check similarity of the views in translation
    sim_mat = np.zeros((sub_im_num, sub_im_num, sub_im_num, sub_im_num, 2))
    check_all_views = input("Check akk views for similarity? (y/n): ").lower()
    for i in range(0, sub_im_num):
        if check_all_views != "y" and i != 0:
            continue
        for j in range(0, sub_im_num):
            if check_all_views != "y" and j != 0:
                continue
            for k in range(i, sub_im_num):
                for l in range(j, sub_im_num):
                    if i == k and j == l:
                        continue
                    view1 = views[i, j]
                    view2 = views[k, l]
                    cor = correlate2d(view1, view2, mode="same", boundary="fill")
                    max_cor = np.unravel_index(np.argmax(cor), np.shape(cor))
                    max_cor_val = np.max(cor)
                    if len(np.transpose(np.nonzero(cor==max_cor_val))) > 1:
                        print(f"Warning: Multiple maxima found in correlation for views {i+1}-{j+1} and {k+1}-{l+1}. Using the first one.")
                    sim_mat[i, j, k, l, :] = max_cor
    # Display of similarity matrices
    fig_x, ax_x = plt.subplots(nrows=sub_im_num, ncols=sub_im_num, sharex=True, sharey=True)
    fig_y, ax_y = plt.subplots(nrows=sub_im_num, ncols=sub_im_num, sharex=True, sharey=True)
    for idx in range(sub_im_num):    
        for jdx in range(sub_im_num): 
            im_x = ax_x[idx,jdx].imshow(sim_mat[idx, jdx, :, :, 1], cmap="gray")
            im_y = ax_y[idx,jdx].imshow(sim_mat[idx, jdx, :, :, 0], cmap="gray")
            ax_x[idx, jdx].set_title(f"View {idx+1}-{jdx+1} X translation similarity")
            ax_y[idx, jdx].set_title(f"View {idx+1}-{jdx+1} Y translation similarity")
            fig_x.colorbar(im_x, ax=ax_x[idx, jdx], orientation='vertical', fraction=0.02, pad=0.04)
            fig_y.colorbar(im_y, ax=ax_y[idx, jdx], orientation='vertical', fraction=0.02, pad=0.04)
    fig_x.suptitle("X translation similarity")
    fig_y.suptitle("Y translation similarity")
    # plt.show()

    # Show for view (1, 1) the translations and show the average summation
    rect_images = np.zeros((sub_im_num, sub_im_num, d, d))
    plt.figure()
    plt.imshow(views[0, 0], cmap='gray')
    for idx in range(sub_im_num):
        for jdx in range(sub_im_num):
            warp_mat = np.squeeze(np.array([[[1, 0, sim_mat[0, 0, idx, jdx, 1]-d//2], [0, 1, sim_mat[0, 0, idx, jdx, 0]-d//2]]]))
            rect_im = cv.warpAffine(views[idx, jdx], warp_mat, (views[idx, jdx].shape[1], views[idx, jdx].shape[0]))
            rect_images[idx, jdx] = rect_im
            plt.figure()
            plt.imshow(rect_im, cmap="gray")
    # plt.show()

    av_rect = np.mean(rect_images, axis=(0, 1))
    plt.figure()
    plt.imshow(av_rect, cmap="gray")
    plt.show()
    plt.close("all")
