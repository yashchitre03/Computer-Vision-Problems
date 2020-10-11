from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def display_image(arr, title, cmap='gray', grayscale=True):
    if not grayscale:
        arr = arr[:,:,::-1]
    plt.imshow(arr, cmap=cmap)
    plt.title(title)
    plt.show()


def get_accumulator(height, width, E):
    accumulator = np.zeros(shape=(height, width), dtype=np.uint8)
    rho_shift = (E.shape[0]**2 + E.shape[1]**2)**0.5
    max_rho = 2 * rho_shift
    delta_theta = 180 / width
    
    for x, y in np.argwhere(E):
        theta = 0
        for theta_index in range(width):
            # calculating rho
            rho = x*np.cos(theta) + y*np.sin(theta)
            
            # normalizing rho for the accumulator array
            rho = int(((rho + rho_shift) / max_rho) * (height - 1))
            
            # voting in the accumulator array and updating theta
            accumulator[rho, theta_index] += 1
            theta += delta_theta
            
    return accumulator


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
display_image(inputImg, 'INPUT IMAGE', grayscale=False)

# apply canny edge detector
E = cv.Canny(image=inputImg, threshold1=100, threshold2=200)
display_image(E, 'Canny result')

# get the hough voting matrix
acc = get_accumulator(1000, 1000, E)
display_image(acc, 'Accumulator array')

