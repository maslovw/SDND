# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The Project
---

The goals / steps of this project are the following:

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

### 1. Observing data
 
Udacity provides two data sets of [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) 
and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images.

![examples_vehicle_non-vehicle](data_example)

All images are 64x64 pixels, in total there are 8792 images of vehicles and 8968 non-vehicle .png images

### 2. Image preprocessing
Some of the images are a bit blury or light distributed quite oddly, which makes it harder to 
recognize a vehicle on the picture

There are many ways how to adjust picture and partly get rid of shadows.
I chose matlabs feature [imadjust](https://stackoverflow.com/a/44529776/4875690)

![imadjust_example][imadjust_example]

A square root of the image normalizes it and gets uniform brightness, reducing the effect of shadows

![prep_image][prep_image]

### 3. Extracting features

#### 3.1. Color space
Before trying out the HOG, I decided to see, what a picture of a vehicle looks like in different
color spaces

![color_spaces][color_spaces]

As advised I took YCrCb color space for classification

![YCrCb_example][ycrcb_example]

#### 3.2. HOG

[Histogram of Oriented Gradients (HOG)](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html)

![hog_example][hog_exapmle]

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
https://render.githubusercontent.com/view/ipynb?commit=abb99be7b6e0741d41fbd7e4de5ae57ea884d992&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6d61736c6f76772f53444e442f616262393962653762366530373431643431666264376534646535616535376561383834643939322f56656869636c65446574656374696f6e2f76656869636c655f646574656374696f6e2e6970796e62&nwo=maslovw%2FSDND&path=VehicleDetection%2Fvehicle_detection.ipynb&repository_id=93343077&repository_type=Repository#Extracting-features


