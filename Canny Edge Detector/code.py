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
    
    op = gaussianSmoothing(lenaImg, 3)
    
    cv.imshow('op', op)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
def gaussianSmoothing(ip, size, sigma=1):
    s = 2 * sigma**2
    denom = 1 / (np.pi * s)
    kernel = np.full((size, size), denom)

    x, y = np.mgrid[-(size//2): size//2+1, -(size//2): size//2+1]
    kernel *= np.exp(-(x**2 + y**2) / s)
    kernel /= np.abs(kernel).sum()
    
    
    # add padding to the input images to preserve size
    sidePadding = (size - 1) // 2
    padding = (sidePadding, sidePadding)
    paddedIp = np.pad(ip, (padding, padding), 'constant')
    
    op = np.empty(ip.shape)
    for x in range(ip.shape[0]):
        for y in range(ip.shape[1]):
            dx = x + kernel.shape[0]
            dy = y + kernel.shape[1]
            op[x, y] = np.sum(paddedIp[x:dx, y:dy] * kernel)
              
    np.clip(op, 0, 255, op)
    op = op.astype('uint8')
    return op


if __name__ == '__main__':
    main()