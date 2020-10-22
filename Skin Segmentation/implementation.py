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

for file in glob(str(train_path / '*.jpg')):
    image = cv.imread(file, -1)
    HSV_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    for row in HSV_image:
        for H, S, _ in row:
            h_index = H // VALUES_PER_INDEX
            s_index = S // VALUES_PER_INDEX
            histogram[h_index, s_index] += 1
                        
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
    plt.show()
    
    cv.imwrite(filename=str(op_path / f'op_{file_no}.bmp'), img=res)