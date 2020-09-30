from pathlib import Path
import numpy as np
from numpy import pi as PI
import cv2 as cv
import matplotlib.pyplot as plt


def main():
    
    # add paths to the input images
    path = Path.cwd() / 'res'
    inputRes = path / 'input'
    outputRes = path / 'output'
    lenaPath = str(inputRes / 'lena_gray.png')
    testPath = str(inputRes / 'test.png')
    
    # read images
    lenaImg = cv.imread(lenaPath, -1)
    lenaSize = lenaImg.shape
    testImg = cv.imread(testPath, -1)
    testSize = testImg.shape
    
    # check whether images are read correctly
    plt.imshow(lenaImg, cmap='gray')
    plt.title("lena")
    plt.show()
    plt.imshow(testImg, cmap='gray')
    plt.title("test")
    plt.show()
    
    # 1. Gaussian Smoothing
    S = gaussianSmoothing(testImg)
    plt.imshow(S, cmap='gray')
    plt.title("Smoothed")
    plt.show()
    
    # 2. Calculating image gradient
    mag, theta = imageGradient(S)
        
    # 3. Suppressing Nonmaxima
    mag = nonMaximaSuppress(mag, theta)
    plt.imshow(mag, cmap='gray')
    plt.title("suppressed Mag")
    plt.show()
    
    # 4. Thresholding and Edge Linking
    mag = hysteresisThreshold(mag)
    plt.imshow(mag, cmap='gray')
    plt.title("Hysteresis Thresholding")
    plt.show()
    E = edgeLinking(mag)
    plt.imshow(E, cmap='gray')
    plt.title("Edge Linking")
    plt.show()


def convolution(ip, kernel):
    '''
    applies the kernel on the image

    Args:
        ip (numpy array): input image.
        ipShape (tuple): input array shape.
        kernel (numpy array): image filter.

    Returns:
        opImg (numpy array): output image.

    '''
    kernel = np.flip(kernel)
    
    # add padding to the input images to preserve size
    sidePadding = (kernel.shape[0] - 1) // 2
    padding = (sidePadding, sidePadding)
    paddedIp = np.pad(ip, (padding, padding), 'constant')
    
    op = np.empty(ip.shape)
    for x in range(ip.shape[0]):
        for y in range(ip.shape[1]):
            dx = x + kernel.shape[0]
            dy = y + kernel.shape[1]
            op[x, y] = np.sum(paddedIp[x:dx, y:dy] * kernel)
              
    return op


def gaussianSmoothing(ip, size=7, sigma=2):
    s = 2 * sigma**2
    denom = 1 / (np.pi * s)
    kernel = np.full((size, size), denom)

    x, y = np.mgrid[-(size//2): size//2+1, -(size//2): size//2+1]
    kernel *= np.exp(-(x**2 + y**2) / s)
    kernel /= np.abs(kernel).sum()
    
    return convolution(ip, kernel)


def imageGradient(ip):
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    
    hor_grad = convolution(ip, sobel)
    ver_grad = convolution(ip, sobel.T)
    
    plt.imshow(hor_grad, cmap='gray')
    plt.title("hor")
    plt.show()
    plt.imshow(ver_grad, cmap='gray')
    plt.title("ver")
    plt.show()

    magnitude = np.sqrt(np.square(hor_grad) + np.square(ver_grad))
    magnitude *= 255 / magnitude.max()
    theta = np.arctan2(ver_grad, hor_grad)
      
    plt.imshow(magnitude, cmap='gray')
    plt.title("mag")
    plt.show()
    plt.imshow(theta)
    plt.title("theta")
    plt.show()

    return magnitude, theta


def nonMaximaSuppress(mag, theta):
    suppressedMag = np.zeros(mag.shape)
    h, w = theta.shape[0], theta.shape[1]
    
    for x in range(h):
        for y in range(w):
            angle = theta[x, y]
            
            if (PI/8 < angle <= 3*PI/8) or (-5*PI/8 <= angle < -7*PI/8):
                firstNeigh = mag[x+1, y-1] if (x < h-1 and y > 0) else 0
                secondNeigh = mag[x-1, y+1] if (x > 0 and y < w-1) else 0
                
            elif (3*PI/8 < angle <= 5*PI/8) or (-3*PI/8 <= angle < -5*PI/8):
                firstNeigh = mag[x-1, y] if (x > 0) else 0
                secondNeigh = mag[x+1, y] if (x < h-1) else 0
                
            elif (5*PI/8 < angle <= 7*PI/8) or (-PI/8 <= angle < -3*PI/8):
                firstNeigh = mag[x-1, y-1] if (x > 0 and y > 0) else 0
                secondNeigh = mag[x+1, y+1] if (x < h-1 and y < w-1) else 0
                
            else:
                firstNeigh = mag[x, y-1] if (y > 0) else 0
                secondNeigh = mag[x, y+1] if (y < w-1) else 0
                
            if mag[x, y] >= firstNeigh and mag[x, y] >= secondNeigh:
                suppressedMag[x, y] = mag[x, y]
                
    return suppressedMag


def hysteresisThreshold(mag, low=5, high=10, weak=128, strong=255):
    op = np.zeros(mag.shape)
    op[(mag >= low) & (mag <= high)] = weak
    op[mag > high] = strong
    
    return op


def edgeLinking(mag):
    for x in range(1, mag.shape[0]-1):
        for y in range(1, mag.shape[1]-1):
            if mag[x, y] == 255:
                convertNeighbors(mag, x, y)
            
    mag[mag != 255] = 0
    return mag


def convertNeighbors(arr, x, y):
    for rShift, cShift in moves:
        if 0 < arr[x+rShift, y+cShift] < 255:
            arr[x+rShift, y+cShift] = 255
            convertNeighbors(arr, x+rShift, y+cShift)


if __name__ == '__main__':
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), 
             (-1, -1), (1, 1), (-1, 1), (1, -1)]
    main()