from pathlib import Path
import numpy as np
import cv2 as cv


def main():
    '''
    Preprocesses the input and calls the required functions for filters

    Returns:
        None.

    '''
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
    filterSize = int(input('Enter kernel size: (odd value) '))
    
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
    op = convolution(lenaImg, lenaSize, meanFilter)
    cv.imwrite(lenaPath_mean, op)
    op = convolution(lenaImg, lenaSize, gaussFilter)
    cv.imwrite(lenaPath_gauss, op)
    op = convolution(lenaImg, lenaSize, sharpenFilter)
    cv.imwrite(lenaPath_sharpen, op)
    
    # # apply filters to the 'art.png' image
    op = convolution(artImg, artSize, meanFilter)
    cv.imwrite(artPath_con_mean, op)
    op = correlation(artImg, artSize, meanFilter)
    cv.imwrite(artPath_corr_mean, op)
    op = applyMedianKernel(artImg, artSize, filterSize)
    cv.imwrite(artPath_median, op)
    
    
def getMeanKernel(size):
    '''
    generates the mean kernel/filter with a given size

    Args:
        size (int): size of kernel.

    Returns:
        kernel (numpy array): mean kernel.

    '''
    kernel = np.ones((size, size))
    kernel *= 1/(size**2)
    return kernel

def getGaussianKernel(size, sigma=1):
    '''
    generates the gaussian kernel/filter with a given size

    Args:
        size (int): size of kernel.
        sigma (int, optional): parameter for gaussian curve. Defaults to 1.

    Returns:
        kernel (numpy array): gaussian kernel.

    '''
    s = 2 * sigma**2
    denom = 1 / (np.pi * s)
    kernel = np.full((size, size), denom)

    x, y = np.mgrid[-(size//2): size//2+1, -(size//2): size//2+1]
    kernel *= np.exp(-(x**2 + y**2) / s)
    
    kernel /= np.abs(kernel).sum()
    
    return kernel

def getSharpenKernel(size, alpha=3):
    '''
    generates the sharpening kernel given a size

    Args:
        size (int): size of kernel.
        alpha (int, optional): parameter for amount of sharpening. Defaults to 3.

    Returns:
        kernel (numpy array): sharpening kernel.

    '''
    center = np.zeros((size, size))
    center[size//2, size//2] = alpha + 1
    
    meanFilter = getMeanKernel(size)
    
    kernel = center -  alpha * meanFilter
    return kernel

def convolution(ip, ipShape, kernel):
    '''
    flips the kernel and calls applyKernel()

    Args:
        ip (numpy array): input image.
        ipShape (tuple): input array shape.
        kernel (numpy array): image filter.

    Returns:
        TYPE: output image.

    '''
    kernel = np.flip(kernel)
    return applyKernel(ip, ipShape, kernel)

def correlation(ip, ipShape, kernel):
    '''
    calls applyKernel

    Args:
        ip (numpy array): input image.
        ipShape (tuple): input array shape.
        kernel (numpy array): image filter.

    Returns:
        TYPE: output image.

    '''
    return applyKernel(ip, ipShape, kernel)

def applyKernel(ip, ipShape, kernel):
    '''
    applies the kernel on the image

    Args:
        ip (numpy array): input image.
        ipShape (tuple): input array shape.
        kernel (numpy array): image filter.

    Returns:
        opImg (numpy array): output image.

    '''
    opImg = np.empty(ipShape)
    for color in range(3):
        for x in range(ipShape[0]):
            for y in range(ipShape[1]):
                dx = x + kernel.shape[0]
                dy = y + kernel.shape[1]
                opImg[x, y, color] = np.sum(ip[x:dx, y:dy, color] * kernel)
              
    opImg = clip(opImg)
    return opImg

def applyMedianKernel(ip, ipShape, size):
    '''
    applies median kernel on the input image

    Args:
        ip (numpy array): input image.
        ipShape (tuple): input image shape.
        size (int): kernel size.

    Returns:
        opImg (numpy array): output image.

    '''
    opImg = np.empty(ipShape)
    for color in range(3):
        for x in range(ipShape[0]):
            for y in range(ipShape[1]):
                dx = x + size
                dy = y + size
                opImg[x, y, color] = np.median(ip[x:dx, y:dy, color])
    
    opImg = clip(opImg)               
    return opImg  

def clip(ip):
    '''
    clips the output image values between 0 and 255

    Args:
        ip (numpy array): input image.

    Returns:
        ip (numpy array): clipped image.

    '''
    np.clip(ip, 0, 255, ip)
    ip = ip.astype('uint8')   
    return ip
    
if __name__ == '__main__':
    main()