from pathlib import Path, os
import numpy as np
from glob import glob
import cv2 as cv
from matplotlib import pyplot as plt


class Image:
    sift = cv.SIFT_create()
    
    def __init__(self, img, true_label):
        self.img = img
        self.true_label = true_label
        self.pred_label = None
        self.keypoints, self.descriptors = self.feature_detector(img)
        self.histogram = None
        
    @classmethod
    def feature_detector(cls, img):
        return cls.sift.detectAndCompute(img, None)
    
    def visualize_features(self):
        copy = self.img.copy()
        copy = cv.drawKeypoints(self.img, self.keypoints, copy, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.imshow(self.img, 'gray')
        plt.title(f'Image from class {self.true_label}')
        plt.axis('off')
        plt.show()
        
        plt.imshow(copy)
        plt.title('Image with feature detection')
        plt.axis('off')
        plt.show()


root_path = Path.cwd()
train_path = root_path / 'data' / 'train' / '*'
validation_path = root_path / 'data' / 'validation' / '*'

train_data = []
for folder in glob(str(train_path)):
    label = folder.split(os.sep)[-1]
    for file in glob(folder + '/*.jpg'):
        image = cv.imread(file, -1)
        img_obj = Image(img=image, true_label=label)
        train_data.append(img_obj)
 
for img in train_data[0:500:100]:
    img.visualize_features()