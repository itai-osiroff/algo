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
    
    normalized_array = (image_array - min_val) / (max_val - min_val) * 255
    normalized_image = Image.fromarray(normalized_array.astype('uint8'))
    
    return normalized_image

if __name__ == "__main__":
    images=[]
    for f in os.listdir('images'):
        image_path = os.path.join('images', f)
        image = load_image(image_path)
        if image:
            images.append(image)
            print(f"Successfully loaded image: {f}")
        else:
            print(f"Failed to load image: {f}")
    for i, im in enumerate(images):
        norm_im = min_max_normalize(im).rotate(-5)
        images[i] = norm_im
        # norm_im.show()
    circles_chosen = None
    for i, im in enumerate(images):
        cimg = cv.cvtColor(np.array(im),cv.COLOR_GRAY2BGR)
        circles = cv.HoughCircles(np.array(im), cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=50, maxRadius=120)
        circles = np.int64(np.around(circles))
        for i in circles[0,:2]:
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
            break
        plt.close("all")
    # cv.destroyAllWindows()
    if circles_chosen is None:
        circles_chosen = circles

    # Let's use the first 2 circles for one of the images found as the basis for the other circles assuming a square grid
    windows = circles_chosen[0, :2]
    d = int(np.linalg.norm(windows[0][:2] - windows[1][:2])/np.sqrt(2))
    sub_im_num = 3
    d = np.int_(np.floor(1.1*2*np.max(windows[:, 2])))
    views = np.zeros((sub_im_num, sub_im_num, d, d))
    norm_im = np.array(norm_im)
    for i in range(0, sub_im_num):
        for j in range(0, sub_im_num):
            x = int(windows[0][0] + (i-sub_im_num//2) * d)
            y = int(windows[0][1] + (j-sub_im_num//2) * d)
            if x - (d//2) < 0 or y - (d//2) < 0 or x + (d//2) >= np.shape(norm_im)[0] or y + (d//2) >= np.shape(norm_im)[1]:
                print(f"Skipping view {i+1}-{j+1} due to out of bounds coordinates.")
                continue
            views[i, j, :, :] = np.array(norm_im)[y-(d//2):y+(d//2)+1, x-d//2:x+d//2+1]

    # for i in range(0, sub_im_num):
    #     for j in range(0, sub_im_num):    
    #         cv.namedWindow(f'view {i+1}-{j+1}', cv.WINDOW_NORMAL)
    #         view_for_imshow = np.array(min_max_normalize(Image.fromarray(views[i, j])))
    #         cv.imshow(f'view {i+1}-{j+1}', view_for_imshow)
    #         cv.waitKey(0)
    #         cv.destroyAllWindows()
    
    # Check similarity of the views in translation
    sim_mat = np.zeros((sub_im_num, sub_im_num, sub_im_num, sub_im_num, 2))
    for i in range(0, sub_im_num):
        for j in range(0, sub_im_num):
            for k in range(i, sub_im_num):
                for l in range(j, sub_im_num):
                    if i == k and j == l:
                        continue
                    view1 = views[i, j]
                    view2 = views[k, l]
                    cor = correlate2d(view1, view2, mode="same", boundary="fill")
                    max_cor = np.unravel_index(np.argmax(cor), np.shape(cor))
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
    plt.show()

    # Show for view (1, 1) the translations and show the average summation
    rect_images = np.zeros((sub_im_num, sub_im_num, d, d))
    plt.figure()
    plt.imshow(views[0, 0], cmap='gray')
    for idx in range(sub_im_num):
        for jdx in range(sub_im_num):
            warp_mat = np.squeeze(np.array([[[1, 0, sim_mat[0, 0, idx, jdx, 1]], [0, 1, sim_mat[0, 0, idx, jdx, 0]]]]))
            rect_im = cv.warpAffine(views[idx, jdx], warp_mat, (views[idx, jdx].shape[1], views[idx, jdx].shape[0]))
            rect_images[idx, jdx] = rect_im
            plt.figure()
            plt.imshow(rect_im, cmap="gray")
    plt.show()

    av_rect = np.mean(rect_images, axis=(0, 1))
    plt.figure()
    plt.imshow(av_rect, cmap="gray")
    plt.show()
