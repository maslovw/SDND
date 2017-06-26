---
### Writeup / README

Here is a link to my [project code](https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration
#### Training set visualisation
![visual](https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/misc/training_set_visualisation.png)


#### Basic summary

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 42

#### Distribution of the data

The amount of some labels is much higher than amount of other labels
Here is an exploratory visualization of the data set. It is a bar chart showing how the training set distributed
![training_set](https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/misc/training_set_distribution.png)
The minimum is 'Go straight or left' sign: 180 times(0.52%)
The maximum got the 'Speed limit (50km/h)' sign with 2010 different pictures, which is 5.78%

### Design and Test a Model Architecture
### 1. Multi-column Deep CNN
Before I started the project, I decided to check out what's already done for this data set.
And I liked the approach of [Multi-column DNN](https://arxiv.org/abs/1202.2745), which I decided to try.
Here's the architecture of my multi-column DNN
![DNN](https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/misc/DNNstructure.png)

#### Image preprocessing
As described in the paper, each of parallen deep CNNs take preprocessed picture:
![image_preprocessing](https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/misc/img_preproc.png)
1. source image
2. [imadjust](https://stackoverflow.com/questions/31647274/is-there-any-function-equivalent-to-matlabs-imadjust-in-opencv-with-c/31650693#31650693)
3. [equilizedHist on HSV](http://opencv-srf.blogspot.de/2013/08/histogram-equalization.html)
4. [adaptHisteq](https://de.mathworks.com/help/images/ref/adapthisteq.html)

#### Results
The result was impressive, as I didn't try any other model.
[Here's the link to see how it worked](http://htmlpreview.github.io/?https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/MCDNN/WithoutSourceImg.html)
On my laptop training the model with validating it took ~28sec/epoch with 3 DNNs (and 36sec with 4 DNNs, batch_size=128)
The best result was:
`loss: 0.0061 - acc: 0.9984 - val_loss: 0.0883 - val_acc: 0.9798`

### LeNet

After being all excited about this architecture with parrallel CNNs, I decided to try another architecture, which was recomended.

#### Image preprocessing

Only pixel normalization by the Lamda layer
`Lambda(lambda x: x/127.5 - 1.)(input_img)`

### Result
[Here's the link to see how it worked](https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/LeNet/Traffic_Sign_Classifier.ipynb)

On my laptop training the model took ~10s/epoch (batch_size=128)
The best result was: 
`loss: 0.0016 - acc: 0.9997 - val_loss: 0.2297 - val_acc: 0.9617`

### LeNet on augmented data
I liked the result, it was not worse then my previous approach, but the training took almost 3 times less time per one epoch, which is good for the performans.

So at the end I wanted to play with Keras generator, and I generated pictures from the training set.
At the end I got 86000 pictures of signs including source training data

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


