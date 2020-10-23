from matplotlib import pyplot as plt
from pathlib import Path
from glob import glob
import numpy as np
import cv2 as cv

main_folder = Path.cwd()
train_path = main_folder / 'Input' / 'train_data'
test_path = main_folder / 'Input' / 'test_data'
op_path = main_folder / 'Results'

VALUES_PER_INDEX = 10
h_size = s_size = (256 // VALUES_PER_INDEX) + 1
histogram = np.zeros(shape=(h_size, s_size))

# part 2
h_sum = s_sum = 0
N = 0

for file in glob(str(train_path / '*.jpg')):
    image = cv.imread(file, -1)
    HSV_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    for row in HSV_image:
        for H, S, _ in row:
            h_index = H // VALUES_PER_INDEX
            s_index = S // VALUES_PER_INDEX
            histogram[h_index, s_index] += 1
            
            # part 2
            h_sum += H
            s_sum += S
            N += 1
                        
histogram = histogram / histogram.max()            

threhsold = histogram.mean()
for file_no, file in enumerate(glob(str(test_path / '*.bmp'))):
    image = cv.imread(file, -1)
    HSV_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    res = np.zeros(shape=image.shape, dtype='uint8')
    
    for i, row in enumerate(HSV_image):
        for j, (H, S, _) in enumerate(row):
            h_index = H // VALUES_PER_INDEX
            s_index = S // VALUES_PER_INDEX
            if histogram[h_index, s_index] > threhsold:
                res[i, j, :] = image[i, j, :]
             
    plt.imshow(res[:, :, ::-1])
    plt.title(f'histogram_{file_no}.bmp')
    plt.show()
    
    cv.imwrite(filename=str(op_path / f'histogram_{file_no}.bmp'), img=res)
    
    
# part 2
E = np.array([[h_sum / N], [s_sum / N]])
C = np.zeros(shape=(2, 2), dtype='float32')

for file in glob(str(train_path / '*.jpg')):
    image = cv.imread(file, -1)
    HSV_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    for row in HSV_image:
        for H, S, _ in row:
            h_prime = H - E[0][0]
            s_prime = S - E[1][0]
            C += np.array([[h_prime**2, h_prime*s_prime], [s_prime*h_prime, s_prime**2]])
            
C /= N
threhsold = 0.01
for file_no, file in enumerate(glob(str(test_path / '*.bmp'))):
    image = cv.imread(file, -1)
    HSV_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    res = np.zeros(shape=image.shape, dtype='uint8')
    
    for i, row in enumerate(HSV_image):
        for j, (H, S, _) in enumerate(row):
            x = np.array([[H], [S]])
            likelihood = np.exp(-0.5 * (x - E).T @ C**-1 @ (x - E))
            if likelihood > threhsold:
                res[i, j, :] = image[i, j, :]
             
    plt.imshow(res[:, :, ::-1])
    plt.title(f'gaussian_{file_no}.bmp')
    plt.show()
    
    cv.imwrite(filename=str(op_path / f'gaussian_{file_no}.bmp'), img=res)
