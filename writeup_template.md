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
[image1]: ./output_images/test1_sliding.jpg
[image2]: ./output_images/test3_sliding.jpg
[image3]: ./output_images/test6_sliding.jpg
[image4]: ./output_images/test1_heatmap.jpg
[image5]: ./output_images/test1_final.jpg
[image6]: ./output_images/test3_heatmap.jpg
[image7]: ./output_images/test3_final.jpg
[image8]: ./output_images/test6_heatmap.jpg
[image9]: ./output_images/test6_final.jpg
[video1]: ./output_images/project_video_labeled.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation. Please refer to the submitted IPython notebook titled "Video Pipeline-Submission" for any references to code.  

---
###Histogram of Oriented Gradients (HOG) & Color Histograms

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I chose to use HOG and color histograms to construct a feature for my classifier. I will describe both processes below.

I extracted HOG features from the training images (and later from the project images) using the function "get_hog_features" in the IPython notebook (2nd cell). This function is called via the "extract_features" (5th cell) when training a classifier and the "search_windows" (9th cell) function when classifying a candidate image.  I found that the parameters in the table below did a reasonable job of correctly classifying features when combined with a color histogram. 

|Parameter      |Value|
|---------------|-----|
|orientation    |9    |
|pixels per cell|8    |
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

####2. Explain how you settled on your final choice of HOG and color histogram parameters.

I settled on my final choice for HOG and color histogram parameters via experimentation (brute force...). In order to do this I would define a set of parameters, train my linear SVM, note the test accuracy and visually inspect the result on a test image, and repeat. Since the end result of this project relies on a heat map to aggregate multiple detections, I felt that a visual inspection of a test image was a better way to judge accuracy than the test accuracy. However, the test accuracy does provide good guidance in general.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG and color histogram features.

The code block that I used to train my linear SVM is in (cell 14) of the IPython notebook. I began by loading the vehicle and non-vehicle images provided and running them through my "extract_features" function (cell 5) to create a feature vector for each image. I then used sklearn.preprocessing.StandardScalar() to scale the images to a zero mean and unit variance to prevent one portion of the feature (HOG or color histogram) from overpowering the other when classifying. Next, I split the data into 90% training and 10% test data. I chose to use a larger portion of the dataset for training because the ultimate goal of the project is to generate a heat map based upon multiple detections, which I felt was better represented by visual inspection than through test data scores. I then finally trained my classifier. I experimented with nonlinear and linear SVMs and like the accuracy/performance of the linear SVM the most.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented my sliding window search in (cell 6) of the IPython notebook.

I chose to bound my search so that it started 300 pixels from the top of the image, which roughly corresponds with the horizon plus some buffer. By focusing on the region of the image that contained the road, this reduced false positives and improved the speed of the algorithm. I also chose to use five sizes of windows: 48, 96, 144, 196, and 250 pixels. Finally, I used an overlap of 75%, meaning that each window will overlap the previous in the x and y direction by 75%.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As discussed before, I ended up with a linear SVM utilizing a color histogram and 3 channel YCrCb HOG to build the feature vector. I relied on the ability of the classifier to find multiple hits on a vehicle and relied heavily on the heat map as a filter to identify the final shape. I think my classifier could be improved by modifying the HOG and color histogram parameters to define a more distinct feature vector and by investigating the use of other classifiers. This would also allow me to reduce my reliance on the heat map. I also think more training data would improve classifier accuracy.

###Here are some examples:

![alt text][image1]
![alt text][image2]
![alt text][image3]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_labeled.mp4). The video pipeline function, "process_image", can be found in (cell 13)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used heatmaps to filter out false positives and combine overlapping bounding boxes. See the bottom of the video pipeline, ("process_image", cell 13) along with the functions "add_heat" (cell 10), "apply_threshold" (cell 11) and draw_labeled_bboxes (cell 12) for the implementation. 

The first step is to create a heatmap from the boxes identified in the sliding windows search. The heatmap is created by adding (1) to the location of each pixel for each box that overlaps with it.

Next, I apply a threshold to the heatmap to reduce false positives. Any pixel in the heatmap which has a value below the threshold will be set to zero. I found a threshold of (3) to work well for my classifier.

Finally, I draw a bounding bow around the extent of each blob that remains in the heatmap. If all goes well, these boxes will match the vehicles in the frame.

###Here are a few example heatmaps and their corresponding bounding boxes:

![alt text][image1]
![alt text][image4]
![alt text][image5]
![alt text][image2]
![alt text][image6]
![alt text][image7]
![alt text][image3]
![alt text][image8]
![alt text][image9]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

To be completely honest, I think there is lots of room for improvement in this project. I wasn't able to explore the different classifiers presented in the lecture as much as I would have liked to. My pipeline does a solid job of identifying large vehicles, but also has too many false positives in shadows and trees. I think I could improve performance by further exploring the tools presented in the lectures and taking a second look at how I select the training data.

First, I would like to do a deeper study of the classifiers and parameters. I think that I would be able to reduce the number of false positives by spending more time investigating the different classifiers and how the HOG and color histogram parameters affect the training results.

Second, I would like to modify my sliding window search. I can adjust the search space to reflect the fact that the road narrows at the horizon, and can scale the size of the sliding window to match the perceived size of the vehicles as they approach the horizon. This should help reduce false positives and improve execution time.

Finally, as mentioned earlier I would like to reduce the influence of the heat map in the algorithm. I can more comfortably do this once I have a stronger classifier.

Beyond that, I think that more training data would improve the robustness of the pipeline. I noticed that the classifier particularly struggles with white cars, perhaps there are certain vehicle characteristics that are underrepresented by the training data provided.


