# Computer Vision Projects

Contains 5 different Machine learning tasks in the Computer Vision field.

## Filters
Various filters are applied on sample images, and the results are analyzed.
1. Preprocessing
Both the images, ‘lena.png’ and ‘art.png’ are read and padded with zeros so that the output images won’t lose their size (shape).
2. Mean Kernel
A kernel of the given size is created of constant ‘1’ value. Each element in the 2D array is divided by the total number of elements in the array, to get average value for each (hence, mean filter/kernel).
3. Gaussian Kernel
X and Y values are generated with a mid of 0. This is used to generate a 2D gaussian kernel. This kernel has a peak in the center, and slowly gets smaller as we go towards the edges, with smallest values at the corners (one can imagine this like a 2D gaussian curve viewed from above). These values may not sum exactly to one, so they are normalized accordingly.
4. Sharpening Kernel
To get a sharpened image, we need to subtract the smoothened image from the original image to get a high pass filter. When this filter is added back into the original image, we essentially get a sharpened image. In the code, we first generate the high pass filter, which will later be convolved with the input image. The alpha values in the code can be adjusted according to how sharp an output is required.
5. Convolution
Here, we flip the kernel obtained from the previous steps, and apply it on the image on each color array (RGB) one by one. The result just computes the weighted sum of the input (where the weights are the kernel values).
6. Cross-correlation
This is the same as convolution, but the kernel is not flipped.
7. Median Kernel
The median values of the input array are taken from the elements according to the size of the kernel. Even this is applied for each color array separately.

## Canny Edge Detector
Edge detection is performed on the image using the Canny algorithm pipeline

### Gaussian Smoothing
* The kernel sizes tried are 3 and 5. Kernel size 5 and sigma 5 produces a smoother image than kernel size 3 and sigma 1.
* In gaussian smoothing, different sigma values make more impact on the results. So, the tests have been run on sigma values of 1 and 5 for both the images.
* As can be seen, for `sigma = 1` we have lesser smoothing. Hence, the resulting images not only have more detail but also have some unwanted noise.
* Some of the edges in the hat of the **lena.png** image is connected for **sigma = 1** but they seem disconnected for the higher sigma value. So, we lost some edge information when the sigma was increased.
* But, this is not necessarily bad as lower noise is also observed in the results where higher sigma was chosen.
* The same effect can be observed in the **test.png** image, higher sigma led to loss of some detailed edges but noise was also reduced.

### Image Gradients
* We can see that the magnitude is high where there is rapid change in the intensity of the image.
* This information is helpful for detecting the edges in the image.
* Also, the magnitude is effectively telling us the direction of the gradient. This will help is suppressing the non-maxima values.

### Suppressing Non-maxima
* A single edge is being represented by a thick line in the previous result.
* After non-maxima suppression we have effectively remove this redundancy, and replaced the thick edges with with single pixel width edges. 
* Hence, here the resulting edges are extremely thin and get rid of excess information.

### Hysteresis Thresholding
* Even after getting these edges, we can observe some noise in the results. Hysteresis thresholding is the perfect way to get rid of it.
* The experiments have been run on different thresholds to classiy data into noise, weak edges, and strong edges.
* For the **lena_gray.png** image thresholds of 10 and 20 effectively removed the noise that gaussian smoothing could not remove.
* But when these thresholds were increased to 15 and 30, we can observe some loss of edge information. So it is better to not keep the thresholds too high.
* This does not mean every image can use the same threshold values. For the **test.png** image the thresholds had to be further reduced to get some edges which it would otherwise classify as noise when **lena_test.png** thresholds were used.
* For example, in the **test.png** image, the combination of small sigma (for gaussian kernel) and lower thresholds lead to better retention of edge information than larger sigma and higher thresholds.
* But for **lena_gray.png**, doing the opposite (higher sigma and thresholds) gave better results (reduced noise).
* To summarize, thresholds should not be too low or high, and every image has its own optimal thresholds.

### Edge Linking
* As can be seen in the output, edge linking result is directly dependent upon the thresholding results.
* That is, when thresholds are increased, further edge information is lost. 
* For example, in the **left** region of the **test.png** image, we can observe:
    * For lower thresholds, almost 6 horizontal edges were preserved.
    * But for higher thresholds, even a single horizontal edge can be hardly seen.
* This may be due to the values around strong edges being classified as noise (for higher threhsolds), hence the `edgeLinking` function could not recursively turn it into a strong edge.

## Hough Transform

For line feature detection.

### General Approach
1. Get the input image.
2. Pass it to the Canny edge detector to obtain the edge information.
3. Generate an empty accumulator array.
4. For each point in the edges of the image:
    1. For theta from 0 to pi (180 degrees):
        1. Get rho using the normal form.
        2. Quantize theta and rho values to accumulator indices.
        3. Increment the value at this index in the array.
5. Use thresholding to remove noise from the accumulator array.
6. For each remaining value, find if it is a local maxima and add it to a list accordingly.
7. For each local maxima found, get two points using the line equation and obtain a line between them.
8. Plot all resulting lines on the original image and return the result.

### Result Analysis
1. Running different quantization parameters for the Hough transform yields interesting results.
2. We find changing the quantization drastically changes the final output.
3. We can observe more curves intersecting at the point for smaller accumulator array.
4. As we have more intersections, a lot of points are above the set threshold for finding the local maxima.
5. Hence, we can see more resulting lines on the output image, all of which may not be entirely accurate.
6. Now for the larger accumulator arrays, the values of intersections are not as high as for the previous experiment.
7. Hence, we obtain fewer points as the local maxima, and they map to fewer resulting lines on the image. To compensate, we can lower the threshold when finding the maxima.
8. In short for the smaller matrix, we may count noise as lines, getting lots of lines where some are inaccurate. In the experiment run 1, we can see the algorithm counting the finger in the image as a line.
9. As we increase the accumulator size, for better quantization, we obtain lesser number of lines (but with lower inaccuracies too). At one stage, we start losing information and the edges we as humans categorize as lines are not recognized as lines by the algorithm.
10. So for experiment run 2, only three edges of the page are counted as lines rather than four.
11. Another interesting point to note here is that noise levels in the image affects our final result.
12. For example, in the experiment runs 3 and 5, we used accumulator size 100 by 100. This gives us lines in the image that we expect to be lines.
13. But, for the experiment run 1, even after setting the accumulator size to 200 by 200, we still get some inaccurate information (detecting the finger as a line).
14. Similarly, for experiment runs 4 and 6 setting the accumulator size to 500 by 500 gives us desirable result but for the first image, the same parameters gave one or two lines (therefore I used 400 by 400 size in run 2).
15. To conclude, more noise or larger accumulator array means lower overall votes for the local maxima points. So we need to strike the right balance for the hough voting matrix quantization and take into consideration the the image edge noise.

## Skin Segmentation

Identifying the skin regions in the image.

### Histogram model

* The histogram model has been run 2 times, for a large histogram array and small histogram array.
* Changing the histogram model size plays a significant role in the results we get, which will be discussed at the end.
* But, in general, larger the histogram, the more fine-tuned model we will get. This may sometimes yield poor results (that is, more false negatives), especially for smaller datasets.
* On the other hand, when the histogram is small, we may get better results but this model may also classify noise as skin (that is, more false positives).
* Whatever the histogram size or threshold values are fixed, the general approach for training a histogram model is:
    1. Initialize an empty histogram with certain size.
    2. Take a training image and convert it to HSV values in order to extract the hue and saturation.
    3. Map the hue and saturation pair (one for each pixel in the image) to a corresponding index in the histogram model.
    4. Increment the value at that indices by 1.
    5. Repeat steps 2, 3, and 4 for each training image.
    6. Normalize the histogram values between 0 and 1.
* Now, that we have trained a histogram model, we can test it on unseen images.
* We basically have a model that will tell us at what hue and saturation levels were the most skin pixels noticed.
* We can use thresholding to remove the smaller values, because these could be noise from the training images and not actual skin pixels.
* To test the unseen images, we can follow these steps:
    1. Take an image and convert it to HSV values in order to extract the hue and saturation.
    2. For each pixel, map the hue and saturation values to our histogram model indices.
    3. If the value at this index is greater than the threshold, classify it as skin else not skin.
    4. Repeat steps 1, 2, and 3 for each testing image.

### Gaussian model

* Gaussian model is based on the probabilistic skin model.
* That is, for a given hue and saturation, the model will compute the probability of that pixel being a skin pixel.
* Then we can threshold this probability to get the final answer.
* Gaussian models are more computationally intensive than the histogram model. This is because instead of just voting on certain hue and saturation, here we need to compute the expected values and the covariance matrix from the data.
* But the gaussian model uses much less memory than the histogram model. Here, we don't need huge 2D matrix to store the model.
* We only need to store 6 parameters (2 for mean and 4 for the covariance matrix).
* Hence, gaussian model is also called a parametric model, while histogram is called a non-parametric model.
* Due to its probabilistic nature, gaussian model may sometimes yield better results.
* The general approach is,
    1. Go through all the images and compute the mean values and the covariance matrix.
    2. Use these values on the test images to get the probability.
    3. Classify depending on the probability.
* The likelihood or probability formula is a simple formula based on the hue, saturation, mean, and covariance between hue and saturation.

### Result analysis

* As we can observe, both the models have done a good job in classifying the skin parts from the non-skin parts of the image.
* For the bigger histogram model, some pixels in the middle of the hand and on the fingers were classified as non-skin pixels.
* We could have improved this by doing either of the two things:
    1. Increase the training dataset size.
    2. Decrease the threshold.
* Both the methods will improve the result, but they come with certain caveats.
* Increasing the training size means more time invested in training, which may not be always desireable.
* Decreasing the threshold means risking classifying noise as skin.
* But, if we don't want to choose either of the two options, we have another technique to improve the result: changing the quantization of the histogram model.
* This may not be always best, but will work for our small train and test dataset.
* Previously a hue value, say 100, will map to 100th index, 101 will map to 101st index, and so on.
* But now, multiple values will map and vote for the same index.
* For example, in the second run, all hue and saturation values below 26 will map to the 0th index. Basically we have combined the neighboring values to vote (increment) for the same value.
* In the second run, this improved the results as the pixels in the middle of the hand and on the fingers are now correctly classified as skin pixels, without including a lot of noise.
* Similarly, for the gaussian model, we have achieved excellent results. The ring on one of the finger is classified as non-skin.
* We can see in the plots, our models have learned that around the low hue range, we are more likely to encounter the skin pixels. And everywhere else, we are least likely to find any skin pixels.
* So, from the visualization, we know that both the models have learned similar knowledge from the training images, but their way of reaching and learning it is different.
* Finally, all the models couldn't smoothly identify the edge of the skin in the image. We can observe jagged lines at the edges in the classification.
* This means the models could not learn enough from the training data. This could be resolved by collecting more images in a wide variety of scenarios and lighting conditions.


## Image Classification

Classifying 10 different image classes using k-means and KNN

### General Approach

* The general idea is to learn the important features from the training images and classify the validation images based on these features.
* We want a bag of visual words representaion just like in text classification for our model.
* So the steps are:
    1. Read all the training images.
    2. Extract the important features from the images using any feature detector (SIFT used in this case).
    3. Convert the features to a descriptor array.
    4. Train Kmeans on this data to get the n cluster centers.
    5. Use these cluster centers to construct a histogram for each image by:
        * Find the cluster for each image feature.
        * Increment the frequency for that cluster for the current image in our histogram.
    6. Now that all the histograms of training images are ready, read the validation data.
    7. Convert the validation image to histogram using previous kmeans model.
    8. Find the k nearest neighbors to this histogram and predict the class as the maximum occurring label in the KNN list.
    9. Compute the accuracy.
    
### Analysis

* Our SIFT feature detector can clearly identify the important features in the images as can be seen in the plots.
* The main points, edges, corners are clearly identified.
* This can also be seen in the kmeans clusters.
* After plotting the cluster center patches, we can observe some horizontal, vertical lines as well as sharp corners and grid-like lines.
* Our kmeans algorithm has clearly worked well.
* Finally, after the predictions and plotting the confusion matrix, we can see that our model does very well.
* Forest and suburb images are classified with high accuracy while the rest also get pretty decent accuracies.
* We can also see from the confusion matrix that our model often confused an Office as a Kitchen. The same also happened between Mountain and Open Country.
* So we can collect more training images for these classes for further better accuracy.
* We ran our model for two Kmeans parameters (number of clusters):
    1. 100 clusters.
    2. 25 clusters.
* The 100 cluster model consistently gave 50% plus accuracy, but when the clusters were dropped to just 25, the accuracy also dropped.
* This shows that lesser number of clusters means our model is also less diverse. While high number of clusters will lead to extremely slow training times.
* So it is better to keep a balance between the kmeans clusters.
* Also we tried the KNN on three different parameters:
    1. k = 1.
    2. k = 5.
    3. k = 10.
* Even though these three values did not budge the validation accuracy much, but on-average k = 5 gave the best accuracy.
* The k value should not be set to too low as the prediction can be affected by noise.
* And higher k value means our model will be too general. Hence, at the end k = 5 was chosen as it gave consistent 50% plus accuracy results.
* Hence, the confusion matrix is plotted for this model (100 clusters and k = 5).
