# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The Project
---

The goals/steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on 
a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as
 well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a
 selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later 
 implement on full project_video.mp4) and create a heat map of recurring detections frame
 by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Introduction
The goal of this project is to find cars on the frame, draw boundary boxes around the cars and track them 
on the video stream. 

It is required to use HOG (Histogram of Oriented Gradients) feature extraction and Linear SVM classifier
(or any other classifier)


## Implementation
Code itself and some off description is in the `vehicle_detection` jupyter notebook


[//]: # (Image References)
[data_example]: ./images/data_example.png
[imadjust_example]: ./images/imadjust_example.png
[prep_image]: ./images/prep_image.png
[color_spaces]: ./images/color_spaces.png
[hog_example]: ./images/hog_example.png
[ycrcb_example]: ./images/ycrcb_example.png
[scaling_data]: ./images/scaling_data.png
[cls_fp]: ./images/cls_fp.png
[cnn_heatmap]: ./images/heatmap_cnn.png
[boundary_boxes_svc]: ./images/svc_output.png
[crop]: ./images/crop.png
[sliding_window]: ./images/sliding_window.png
[sliding_hog_window]: ./images/sliding_hog_window.png
[heatmap_svc]: ./images/heatmap_svc.png

[Notebook preview](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html)
### 1. Observing data
 
Udacity provides two data sets of [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) 
and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images.

![examples_vehicle_non-vehicle][data_example]

[code](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Observe-dataset)

All images are 64x64 pixels, in total there are 8792 images of vehicles and 8968 non-vehicle .png images

### 2. Image preprocessing
Some of the images are a bit blurry or light distributed quite oddly, which makes it harder to 
recognize a vehicle on the picture

There are many ways how to adjust picture and partly get rid of shadows.
I chose matlabs feature [imadjust](https://stackoverflow.com/a/44529776/4875690)

[code](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Image-preprocessing)

A square root of the image normalizes it and gets uniform brightness, reducing the effect of shadows

![prep_image][prep_image]

### 3. Extracting features
[code](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Extracting-features)

#### 3.1. Color space
Before trying out the HOG, I decided to see, what a picture of a vehicle looks like in different
color spaces

![color_spaces][color_spaces]

As advised I took YCrCb color space for classification

![YCrCb_example][ycrcb_example]

#### 3.2. HOG

[Histogram of Oriented Gradients (HOG)](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html)

![hog_example][hog_example]

[code](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#HOG-on-LUV-color-space)

For HOG features I choose:
* orientations: 9
* pixels per cells: 8
* cells per block: 1
* image size: 64x64
* color space: ycrcb (all channels)

#### 3.3. Choosing features for classifier

After several hundreds of tries, I decided to stop on:
* Spatial Binning of Color (YCrCb, 16x16)
* Histogram of all YCrCb channels (on 64x64)
* HOG

[code](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Extracting-features)

To get features out of one 64x64 image it takes ~5ms (CoreI7-6820HQ @2.7GHz)

Result of get_features(..) method is a numpy array(shape=(3168,)), which quite a lot of data
for one picture

### 4. Training SVC classifier 

To train the classifer, first I need to prepare the data set:
* [Load](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Loading-training-data-set)
* [Extract all pictures features](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Extracting-features-for-training)
* classifier expects [scaled data](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Scale-the-data)
* [Split dataset into Train and Test sets](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Split-the-dataset-into-training-and-testing-sets)
* [Train LinearSVC](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Training-LinearSVC)

Scaling the data is quite important, otherwise, classifier mostly will see only histograms

![scaling_data.png][scaling_data]

Validating classifier showed
Pretty impressive result

TP  0.48, TN  0.00

FN  0.52, FP  0.00

score: 0.994932432432

However, there are false positives results:

![classification_false_positives][cls_fp]

### 5. Finding cars on image
[code](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Finding-cars-on-image)

To find cars in the image it's required to build sliding window mechanism. 

#### 5.1. Crop picture
To reduce amount of data and cut the sky out, I just cropped the picture

![crop][crop]

#### 5.2. Sliding window
[code](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Sliding-window)

First I implemented simple sliding window generator without using padding.
To take different sizes I resize cropped picture 8 times by factor 1.35 (without using 
original size picture to filter too small windows). 

The sliding window size is 64x64 just like we need for the classification.

To get all sliding windows features from one cropped picture it takes ~5.5 sec! (on my laptop)

![sliding_window][sliding_window]

#### 5.3. Sliding window over HOG

My implementation for HOG sliding window can be found [Here](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Sliding-window-on-HOG) . It's quite simple and probably won't work
with HOGs with parameter 'cells_per_block' > 1

To be able to filter false-positives, it's recommended to build slide windows with overlapping.
As one step I take 1 cell, which is 8 pixels. 

So Sliding Window has window size 64x64 with step 8 pixels on x and y

[Code for extracting](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#calculating-HOG-per-whole-picture,-not-per-single-window)
all features from one cropped image

![sliding_hog_window][sliding_hog_window]

#### 5.4. Heatmap

Because I build sliding window method in the way that I get overlapping windows, 
it's recommended to build heatmap: all the pixels of each window increments by 1

![heatmap_svc][heatmap_svc]

#### 5.5. Boundary boxes
After thresholding heatmap (`heatmap[heatmap<10]') I can build boundary boxes around hot areas,
using `scipy.ndimage.measurements.label(heatmap)` [code](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#Boundary-boxes)

![boundary_boxes_svc][boundary_boxes_svc]

### 6. Video pipeline

[Implementation](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#SVC-pipeline)

To reduce false-positives I use deque(maxlen=10] of thresholded heatmaps, and build stream_heatmap

Boundary boxes are built around thresholded stream_heatmap.

The result of this method can be found [here](https://youtu.be/8hawest3f1U)

It took 38min to build the video. I was not satisfied with the result either. It was my best result using
LinearSVC on HOG

After many tries, I decided to see how much better simple CNN will do.

## 7. CNN classifier
[Link to CNN classification implementation](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#NN-classification)
#### 7.1. CNN structure

|Layer|Description|params|
|:---------------------:|:---------------------------------------------:|:-------:|
|input|64x64x3 bgr picture||
|Lambda x: x/127.5-1|Scaling the picture||
|conv2d|64x64x16|448|
|maxpool|32x32x16||
|conv2d|32x32x32|4640|
|maxpool|16x16x32||
|conv2d|16x16x48|13872|
|maxpool|8x8x48||
|conv2d|8x8x64|27712|
|maxpool|4x4x64||
|conv2d|4x4x96|55392|
|maxpool|2x2x96||
|dense|512|197120|
|dense|256|131328|
|dense|2|514|

Trainable params: 431,026, which is relatively small

Training gave me satisfying result (val_acc ~96%, but val_loss is quite high(20%),
which can give false-positives, but we already have a mechanism to filter that on a video stream)

#### 7.2 CNN heatmap
![Heatmap][cnn_heatmap]

#### 7.3 CNN pipeline
[code](https://maslovw.github.io/SDND/VehicleDetection/vehicle_detection.html#CNN-pipeline)

On my NVIDIA M1000M, it takes about 256ms to classify all the sliding windows on cropped image

[Video on youtube](https://youtu.be/4M_emgQmKIE)

## Summary:

SVC classifier on HOG works, but very slow, and it's really hard to make it robust.
CNN with sliding windows are much faster and simplier, even thow I can improve it to have better results. 

But anyway, now we can use better approach to solve this problem faster and better: YOLO or SSD networks, that produce boundary boxex
itself.

I tryed to play with YOLO on Keras, here's the result https://youtu.be/J9imeiAzQxM

It can work ~23 fps on my laptop, which is pretty impressive.

https://github.com/maslovw/SDND/blob/master/VehicleDetection/Yolo_vehilcle_detection.ipynb

