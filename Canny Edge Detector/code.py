from pathlib import Path
import numpy as np
import cv2 as cv


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
    cv.imshow('lena', lenaImg)
    cv.imshow('test', testImg)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    S = gaussianSmoothing(lenaImg)
    
    cv.imshow('smoothed', S)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    mag, theta = imageGradient(S)
    

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
              
    return normalize(op)

def normalize(arr):
    smallest = arr.min()
    largest = arr.max()
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            arr[x, y] = 255 * ((arr[x, y] - smallest) / (largest - smallest))
    
    arr = arr.astype('uint8')
    return arr

def gaussianSmoothing(ip, size=3, sigma=1.5):
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
    
    magnitude = np.sqrt(np.square(hor_grad) + np.square(ver_grad))
    magnitude = normalize(magnitude)
    theta = np.arctan2(ver_grad, hor_grad)
    theta = normalize(theta)
    
    cv.imshow('hor', hor_grad)
    cv.imshow('ver', ver_grad)
    cv.imshow('mag', magnitude)
    cv.imshow('theta', theta)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return magnitude, theta

if __name__ == '__main__':
    main()