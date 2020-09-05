from pathlib import Path
import numpy as np
import cv2 as cv


def main():
    # add paths to the input images
    path = Path.cwd() / 'res'
    inputRes = path / 'input'
    outputRes = path / 'output'
    lenaPath = str(inputRes / 'lena.png')
    artPath = str(inputRes / 'art.png')
    
    # create path for saving images
    lenaPath_mean = str(outputRes / 'lena_mean.png')
    lenaPath_gauss = str(outputRes / 'lena_gauss.png')
    lenaPath_sharpen = str(outputRes / 'lena_sharpen.png')
    artPath_con_mean = str(outputRes / 'art_con_mean.png')
    artPath_corr_mean = str(outputRes / 'art_corr_mean.png')
    artPath_median = str(outputRes / 'art_median.png')
    
    # read images
    lenaImg = cv.imread(lenaPath)
    lenaSize = lenaImg.shape
    artImg = cv.imread(artPath)
    artSize = artImg.shape
    
    # check whether images are read correctly
    cv.imshow('art', lenaImg)
    cv.imshow('lena', artImg)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # get user input to try different kernel sizes
    filterSize = int(input('Enter kernel size: (preferably odd value) '))
    
    # add padding to the input images to preserve size
    sidePadding = (filterSize - 1) // 2
    padding = (sidePadding, sidePadding)
    lenaImg = np.pad(lenaImg, (padding, padding, (0, 0)), 'constant')
    artImg = np.pad(artImg, (padding, padding, (0, 0)), 'constant')
    
    # get mean and gaussian filters
    meanFilter = getMeanKernel(filterSize)
    gaussFilter = getGaussianKernel(filterSize)
    sharpenFilter = getSharpenKernel(filterSize)
    
    # apply filters to the 'lena.png' image
    op = applyConvolution(lenaImg, lenaSize, meanFilter)
    cv.imwrite(lenaPath_mean, op)
    op = applyConvolution(lenaImg, lenaSize, gaussFilter)
    cv.imwrite(lenaPath_gauss, op)
    op = applyConvolution(lenaImg, lenaSize, sharpenFilter)
    cv.imwrite(lenaPath_sharpen, op)
    
    # apply filters to the 'art.png' image
    op = applyConvolution(artImg, artSize, meanFilter)
    cv.imwrite(artPath_con_mean, op)
    op = applyCorrelation(artImg, artSize, meanFilter)
    cv.imwrite(artPath_corr_mean, op)
    op = applyMedianKernel(artImg, artSize, filterSize)
    cv.imwrite(artPath_median, op)
    
    
def getMeanKernel(size):
    kernel = np.ones((size, size))
    kernel *= 1/(size**2)
    return kernel


def getGaussianKernel(size, sigma=1):
    s = 2 * sigma**2
    denom = 1 / (np.pi * s)
    kernel = np.full((size, size), denom)
    
    x = np.arange(-(size//2), size//2+1)
    y = np.arange(-(size//2), size//2+1)
    xx, yy = np.meshgrid(y, x)
    
    kernel *= np.exp(-(xx**2 + yy**2) / s)
    return kernel


def getSharpenKernel(size):
    alpha = np.zeros((size, size))
    alpha[size//2, size//2] = 2
    
    meanFilter = getGaussianKernel(size)
    
    kernel = alpha -  meanFilter
    return kernel


def applyConvolution(ip, ipShape, kernel):
    kernel = np.flip(kernel)
    opImg = np.empty(ipShape, 'uint8')
    
    for color in range(3):
        for x in range(ipShape[0]):
            for y in range(ipShape[1]):
                dx = x + kernel.shape[0]
                dy = y + kernel.shape[1]
                opImg[x, y, color] = np.sum(ip[x:dx, y:dy, color] * kernel)

    return opImg


def applyCorrelation(ip, ipShape, kernel):
    opImg = np.empty(ipShape, 'uint8')
    
    for color in range(3):
        for x in range(ipShape[0]):
            for y in range(ipShape[1]):
                dx = x + kernel.shape[0]
                dy = y + kernel.shape[1]
                opImg[x, y, color] = np.sum(ip[x:dx, y:dy, color] * kernel)
                
    return opImg


def applyMedianKernel(ip, ipShape, size):
    opImg = np.empty(ipShape, 'uint8')
    
    for color in range(3):
        for x in range(ipShape[0]):
            for y in range(ipShape[1]):
                dx = x + size
                dy = y + size
                opImg[x, y, color] = np.median(ip[x:dx, y:dy, color])
                
    return opImg
    
    
if __name__ == '__main__':
    main()