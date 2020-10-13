from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def display_and_save(arr, title, grayscale=True, save=True):
    
    # save the image in the 'Results' folder
    # this will overwrite when run for each image, all combined results are in 'Submitted Results' folder
    if save:
        cv.imwrite(filename=str(opPath / title) + '.bmp', img=arr)
    
    # handle if images has BGR format
    if not grayscale:
        arr = arr[:,:,::-1]
        
    # display the image
    plt.imshow(arr)
    plt.title(title)
    plt.axis('off')
    plt.show()


def get_accumulator(E):
    accumulator = np.zeros(shape=(HEIGHT, WIDTH), dtype=np.uint8)
    
    for y, x in np.argwhere(E):
        theta = 0
        
        for theta_index in range(WIDTH):
            # calculating rho
            rho = x*np.cos(theta) + y*np.sin(theta)
            
            # normalizing rho for the accumulator array
            rho_index = int((rho + ABS_MAX_RHO) / DELTA_RHO)           
            
            # voting in the accumulator array and updating theta
            accumulator[rho_index, theta_index] += 1
            theta += DELTA_THETA
            
    return accumulator


def get_local_maxima(accumulator, threshold):
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    parameter_space_indices = np.argwhere(accumulator >= threshold)
    local_maximas = []
    
    for rho_index, theta_index in parameter_space_indices:
        maxima = True
        for r_shift, c_shift in neighbors:
            new_rho, new_theta = rho_index + r_shift, theta_index + c_shift
            if 0 <= new_rho < HEIGHT and 0 <= new_theta < WIDTH and \
                accumulator[rho_index, theta_index] > accumulator[new_rho, new_theta]:
                    continue
            else:
                maxima = False
                
        if maxima:
            local_maximas.append((rho_index, theta_index))
            
    return local_maximas


def draw_lines(image, local_maximas):
    for rho_index, theta_index in local_maximas:
        rho = (rho_index * DELTA_RHO) - ABS_MAX_RHO
        theta = theta_index * DELTA_THETA
        alpha = np.cos(theta)
        beta = np.sin(theta)
        x0 = alpha * rho
        y0 = beta * rho
        pt1 = int(x0 + 1000 * (-beta)), int(y0 + 1000 * (alpha))
        pt2 = int(x0 - 1000 * (-beta)), int(y0 - 1000 * (alpha))
        cv.line(image, pt1, pt2, (255, 255, 0))


# get the required file directories
folder = Path.cwd()
ipPath = folder / 'Input'
opPath = folder / 'Results'

# get the input image file path
inputImgpath = str(ipPath / 'input.bmp')
testImgPath = str(ipPath / 'test.bmp')
test2ImgPath = str(ipPath / 'test2.bmp')

# read the input images
inputImg = cv.imread(inputImgpath, -1)
testImg = cv.imread(testImgPath, -1)
test2Img = cv.imread(test2ImgPath, -1)

# check whether the images are read correctly
curImg = inputImg
display_and_save(curImg, 'Input Image', grayscale=False, save=False)

# apply canny edge detector
E = cv.Canny(image=curImg, threshold1=150, threshold2=250)
display_and_save(E, 'Canny Edge Detection')

# set the quantization, rho, and theta parameters
HEIGHT, WIDTH = 180, 180
ABS_MAX_RHO = (curImg.shape[0]**2 + curImg.shape[1]**2)**0.5
DELTA_RHO = (2 * ABS_MAX_RHO) / HEIGHT
DELTA_THETA = np.pi / WIDTH

# get the hough voting matrix
acc = get_accumulator(E)
display_and_save(acc, 'Accumulator')

# find local maximas and plot the resulting lines on the image
local_maximas = get_local_maxima(accumulator=acc, threshold=60)
draw_lines(curImg, local_maximas)
display_and_save(curImg, 'Final Result', grayscale=False)

