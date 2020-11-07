from pathlib import Path, os
import numpy as np
from glob import glob
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import heapq


# class Image:
#     sift = cv.SIFT_create()
    
#     def __init__(self, img, true_label):
#         self.img = img
#         self.true_label = true_label
#         self.pred_label = None
#         self.keypoints, self.descriptors = self.feature_detector(img)
#         self.histogram = None
        
#     @classmethod
#     def feature_detector(cls, img):
#         return cls.sift.detectAndCompute(img, None)
    

# def euclidean_distance(vector1, vector2):
#     dist = 0
#     for val1, val2 in zip(vector1, vector2):
#         dist += (val1 - val2)**2
#     dist = dist**0.5
#     return dist


def visualize_features(image, keypoints, label):
    copy = image.copy()
    copy = cv.drawKeypoints(image, keypoints, copy, 
                            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].imshow(image, 'gray')
    axes[0].set_title('Original image')
    axes[0].axis('off')
    axes[1].imshow(copy)
    axes[1].set_title('After feature detection')
    axes[1].axis('off')
    fig.suptitle(f'Image from class {label}')
    fig.tight_layout()
    plt.show()
    
    
def visualize_dictionary(images, dict_ele):
    plt.figure(figsize=(15, 15))
    for i in range(10):
        for j in range(10):
            cur, keypoint = dict_ele[i*10+j]
            size = 11
            
            start_x = int(keypoint.pt[0] - size / 2.)
            start_y = int(keypoint.pt[1] - size / 2.)
            patch = images[cur][start_y: start_y+size, start_x: start_x+size]
            
            ax = plt.subplot2grid((10, 10), (i, j))
            ax.imshow(patch, 'gray')
            ax.axis('off')
    plt.show()
    

def get_knn(train_histograms, train_labels, val_histogram, k):
    heap = []
    heapq.heapify(heap)
    for histogram, label in zip(train_histograms, train_labels):
        dist = 0
        for val1, val2 in zip(histogram, val_histogram):
            dist += np.square(val1 - val2)
        dist = np.sqrt(dist)
        heapq.heappush(heap, (dist, label))
        
    return [heapq.heappop(heap)[1] for _ in range(k)]


# set the file paths
root_path = Path.cwd()
train_path = root_path / 'data' / 'train' / '*'
val_path = root_path / 'data' / 'validation' / '*'

# create sift descriptor and read the training data
sift = cv.SIFT_create()
train_images, train_labels, train_keypoints, train_descriptors = [], [], [], []
for folder in glob(str(train_path)):
    label = folder.split(os.sep)[-1]
    for file in glob(folder + '/*.jpg'):
        image = cv.imread(file, -1)
        kp, des = sift.detectAndCompute(image, None)

        train_images.append(image)
        train_labels.append(label)
        train_keypoints.append(kp)
        train_descriptors.append(des)
 
# visualize the feature detection
for i in range(0, 500, 200):
    visualize_features(train_images[i], train_keypoints[i], train_labels[i])
        
# k-means
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters).fit(np.concatenate(train_descriptors))

# visualize the dictionary
closest = pairwise_distances_argmin(kmeans.cluster_centers_, np.concatenate(train_descriptors))
closest.sort()
i = j = 0
dict_ele = []
for image_index in range(len(train_keypoints)):
    for keypoint in train_keypoints[image_index]:
        if j < len(closest) and i == closest[j]:
            dict_ele.append((image_index, keypoint))
            j += 1
        i += 1
visualize_dictionary(train_images, dict_ele)

# build training histograms
train_histograms = []
for des in train_descriptors:
    predictions = kmeans.predict(des)
    histogram, _ = np.histogram(predictions, bins=n_clusters)
    train_histograms.append(histogram)
    
# normalize the histograms
train_histograms = np.array(train_histograms)
std = np.std(train_histograms, axis=0)
train_histograms = train_histograms / std

# read the validation data
val_true_labels, val_pred_labels = [], []
for folder in glob(str(val_path)):
    true_label = folder.split(os.sep)[-1]
    for file in glob(folder + '/*.jpg'):
        image = cv.imread(file, -1)
        _, des = sift.detectAndCompute(image, None)
        
        predictions = kmeans.predict(des)
        histogram, _ = np.histogram(predictions, bins=n_clusters)
        histogram = histogram / std
        
        knn = get_knn(train_histograms=train_histograms,
                      train_labels=train_labels,
                      val_histogram=histogram, k=7)
        pred_label = max(set(knn), key=knn.count)
        
        val_true_labels.append(true_label)
        val_pred_labels.append(pred_label)

c = 0
for i, j in zip(val_pred_labels, val_true_labels):
    if i != j: c += 1
print(c)










