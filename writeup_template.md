##P5: Vehicle Detection Submission
###Jeff Fletcher

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation. Please refer to the submitted IPython notbook titled "Video Pipeline-Submission" for any references to code.  

---
###Histogram of Oriented Gradients (HOG) & Color Histograms

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I chose to use HOG and color histograms to construct a feature for my classifier. I will describe both processes below.

I extracted HOG features from the training images (and later from the project images) using the function "get_hog_features" in the IPython notebook (2nd cell). This function is called via the "extract_features" (5th cell) when training a classifier and the "search_windows" (9th cell) function when classifying a candidate image.  I found that the parameters in the table below did a reasonable job of correctly classifying features when combined with a color histogram. 

|Parameter      |Value|
|---------------|-----|
|orientation    |9    |
|pixels per cel |8    |
|cells per block|2    |
|color space    |YCrCb|
|color channels |All  |

The orientation parameter defines the number of bins in which the orientation gradient can be split. For example, with 9 orientations each bin will represent a 40&deg; range. The pixels per cell parameter defines the size of the cell, which in my case is 8 x 8. The cells per block parameter defines the number of cells over which a value will be averaged, which is essentially a filter to smooth the result. I chose to average over a 2 x 2 block of cells. This is done using the YCrCb color space on each channel within the image.

I extracted the color histogram features from the training images (and later from the project images) using the function "color_hist" in the (6th cell) of the IPython notebook. This function is called via the "extract_features" (5th cell) when training a classifier and the "search_windows" (9th cell) function when classifying a candidate image. I chose to reduce the size of the image to 32 x 32 pixels in order to reduce the overhead needed to process the images. I also chose to use 32 histogram bins in the final implementation.

|Parameter      |Value|
|---------------|-----|
|spatial binning|32   |
|histogram bins |32   |

The final feature vector was created by appending the HOG and color histogram features together.



The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG and color histogram parameters.

I settled on my final choice for HOG and color histogram parameters via experimentation (brute force...). In order to do this I would define a set of parameters, train my linear SVM, note the test accuracy and visually inspect the result on a test image, and repeat. Since the end result of this project relies on a heat map to aggregate multiple detections, I felt that a visual inspection of a test image was a better way to judge accuracy than the test accuracy. However, the test accuracy acts as a good guideline.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG and color histogramfeatures.

The code block that I used to train my linear SVM is in (cell 14) of the IPython notebook. I began by loading the vehicle and non-vehicel images provided and running them through my "extract_features" function (cell 5) to create a feature vector for each image. I then scaled the images to prevent one portion of the feature (HOG or color histogram) from overpowering the other when classifying. Next, I split the data into 90% training and 10% test data. I chose to use a larger portion of the dataset for training because the ultimate goal of the project is to generate a heat map based upon multiple detections, which I felt was better represented by visual inspection than through test data scores. I then finally trained my classifier. I experimented with nonlinear and linear SVMs and like the accuracy/performance of the linear SVM the most.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

