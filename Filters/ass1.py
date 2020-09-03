from pathlib import Path
import numpy as np
import cv2 as cv

a = 0
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
    gaussianKernel = getKernel('gaussian', kernelSize, 1)
    sharpenKernel = getKernel('sharpen', kernelSize)
    
    convolution(lenaImg, meanKernel)
    
    convolution(lenaImg, gaussianKernel)
    
    convolution(lenaImg, sharpenKernel)
    
    convolution(artImg, meanKernel)
    
    correlation(artImg, meanKernel)
    
    medianFilter(artImg, kernelSize)
    
    
def getKernel(name, size, sigma=-1):
    if name == 'mean':
        kernel = np.ones((size, size))
        kernel *= 1/(size**2)
    elif name == 'gaussian':
        s = 2 * sigma**2
        denom = 1 / (np.pi * s)
        kernel = np.full((size, size), denom)
        
        x = np.arange(-(size//2), size//2+1)
        y = np.arange(-(size//2), size//2+1)
        xx, yy = np.meshgrid(y, x)
        kernel *= np.exp(-(xx**2 + yy**2) / s)
        print(kernel)
        
        """
        old way to compute gaussian kernel
        for x in range(size):
            for y in range(size):
                kernel[x, y] *= np.exp(-((x-size//2)**2 + (y-size//2)**2) / s)
        """

    elif name == 'sharpen':
        pass
    global a
    a = kernel
    return kernel
    
    
def convolution(ip, kernel):
    pass


def correlation(ip, kernel):
    pass


def medianFilter(ip, size):
    pass
    
    
if __name__ == '__main__':
    main()