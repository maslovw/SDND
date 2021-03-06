# Udacity SDND
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Project **Finding Lane Lines on the Road** 
https://github.com/maslovw/SDND/tree/master/T1.Computer_vision_and_deep_learning/FindingLanes
<img src="T1.Computer_vision_and_deep_learning/FindingLanes/test_images_output/challenge01.jpg" width="480" alt="Combined Image" />
### Overview
When we drive, we use our eyes to decide where to go. 
The lines on the road that show us where the lanes are act as our constant 
reference for where to steer the vehicle. 
Naturally, one of the first things we would like to do in developing a 
self-driving car is to automatically detect lane lines using an algorithm.

In this project we will detect lane lines in images using Python3 and [OpenCV](http://opencv.org/).

## Project **Traffic Sign Recognition**
https://github.com/maslovw/SDND/tree/master/T1.Computer_vision_and_deep_learning/TrafficSignlassifier
<img src = "https://github.com/maslovw/SDND/blob/master/T1.Computer_vision_and_deep_learning/TrafficSignlassifier/misc/training_set_visualisation.png" alt="training set" />
### Overview
The goal of the project is to create a classificator for the German traffic sign data set.

In this project I tryed different classification models using Keras:
* LeNet
* LeNet on augmented data
* Inception
* Multi-column DNN

## Project **Advance Lane Lines on the Road** 
https://github.com/maslovw/SDND/tree/master/T1.Computer_vision_and_deep_learning/Advanced-Lane-Lines
<img src="T1.Computer_vision_and_deep_learning/Advanced-Lane-Lines/output_images/img4.jpg" width="480" alt="Combined Image" />



## Project Behavior Cloning
https://github.com/maslovw/SDND/tree/master/T1.Computer_vision_and_deep_learning/BehavioralCloning

<img src="T1.Computer_vision_and_deep_learning/BehavioralCloning/images/VideoYouTube.png" width="480" alt="Combined Image" />

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

## Project Vehicle Detection
https://github.com/maslovw/SDND/tree/master/T1.Computer_vision_and_deep_learning/VehicleDetection


<img src="T1.Computer_vision_and_deep_learning/VehicleDetection/images/hog_example.png" width="480" alt="Combined Image" />

<img src="T1.Computer_vision_and_deep_learning/VehicleDetection/images/svc_output.png" width="480" alt="Combined Image" />

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
