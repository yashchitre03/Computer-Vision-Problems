from pathlib import Path
import numpy as np
import cv2 as cv


def main():
    # add paths to the input images
    path = Path.cwd() / 'res'
    lenaPath = str(path / 'lena.png')
    artPath = str(path / 'art.png')
    
    # read images
    lenaImg = cv.imread(lenaPath)
    artImg = cv.imread(artPath)
    
    # check whether images are read correctly
    cv.imshow('art', lenaImg)
    cv.imshow('lena', artImg)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # get input to try different kernel sizes
    kernelSize = int(input('Enter kernel size: (preferably odd value) '))
    
    # generate all kernels according to given size
    meanKernel = getKernel('mean', kernelSize)
    gaussianKernel = getKernel('gaussian', kernelSize)
    sharpenKernel = getKernel('sharpen', kernelSize)
    
    convolution(lenaImg, meanKernel)
    
    convolution(lenaImg, gaussianKernel)
    
    convolution(lenaImg, sharpenKernel)
    
    convolution(artImg, meanKernel)
    
    correlation(artImg, meanKernel)
    
    medianFilter(artImg, kernelSize)
    
    
def getKernel(name, size):
    if name == 'mean':
        kernel = np.ones(shape=(size, size), dtype='float32')
        kernel *= 1/(size**2)
        print(kernel)
    elif name == 'gaussian':
        pass
    elif name == 'sharpen':
        pass
    
    return kernel
    
    
def convolution(ip, kernel):
    pass


def correlation(ip, kernel):
    pass


def medianFilter(ip, size):
    pass
    
    
if __name__ == '__main__':
    main()