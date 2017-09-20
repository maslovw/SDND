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

![imadjust_example](imadjust_example)

A square root of the image normalizes it and gets uniform brightness, reducing the effect of shadows

![prep_image](prep_image)

### 3. Extracting features

Before trying out the HOG, I decided to see, what a picture of a vehicle looks like in different
color spaces

![color_spaces][color_spaces]


