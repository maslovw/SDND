---
### Writeup / README

Here is a link to my [project code](https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration
#### Training set visualisation

![visual](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/misc/training_set_visualisation.png)


#### Basic summary

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### Distribution of the data

The amount of some labels is much higher than amount of other labels
Here is an exploratory visualization of the data set. It is a bar chart showing how the training set distributed

![training_set](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/misc/training_set_distribution.png)

The minimum is 'Go straight or left' sign: 180 times(0.52%)
The maximum got the 'Speed limit (50km/h)' sign with 2010 different pictures, which is 5.78%

## Design and Test a Model Architecture
## 1. [Multi-column Deep CNN](http://htmlpreview.github.io/?https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/MCDNN/WithoutSourceImg.html)
Before I started the project, I decided to check out what's already done for this data set.
And I liked the approach of [Multi-column DNN](https://arxiv.org/abs/1202.2745), which I decided to try.
Here's the architecture of my multi-column DNN

![DNN](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/misc/DNNstructure.png)

#### Image preprocessing
As described in the paper, each of parallel deep CNNs take preprocessed picture:

![image_preprocessing](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/misc/img_preproc.png)

1. source image
2. [imadjust](https://stackoverflow.com/questions/31647274/is-there-any-function-equivalent-to-matlabs-imadjust-in-opencv-with-c/31650693#31650693)
3. [equilizedHist on HSV](http://opencv-srf.blogspot.de/2013/08/histogram-equalization.html)
4. [adaptHisteq](https://de.mathworks.com/help/images/ref/adapthisteq.html)

#### Results
The result was impressive, as I didn't try any other model.
[Here's the link to see how it worked](http://htmlpreview.github.io/?https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/MCDNN/WithoutSourceImg.html)
On my laptop training the model with validating took ~28sec/epoch with 3 DNNs (and 36sec with 4 DNNs, batch_size=128)
The best result was:

`loss: 0.0061 - acc: 0.9984 - val_loss: 0.0883 - val_acc: 0.9798`

## 2. [LeNet](https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/LeNet/Traffic_Sign_Classifier.ipynb)

After being all excited about this architecture with parrallel CNNs, I decided to try another architecture, which was recomended.

#### Image preprocessing

Only pixel normalization by the Lamda layer
`Lambda(lambda x: x/127.5 - 1.)(input_img)`

### Result
[Here's the link to see how it worked](https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/LeNet/Traffic_Sign_Classifier.ipynb)

On my laptop training the model took ~10s/epoch (batch_size=128)
The best result was: 

`loss: 0.0016 - acc: 0.9997 - val_loss: 0.2297 - val_acc: 0.9617`

## 3. [LeNet on augmented data](https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/Traffic_Sign_Classifier.ipynb)
I liked the result, it was not that much worse then my previous approach, but the training took almost 3 times less time per one epoch, which is good for the performans.

So at the end I wanted to play with Keras generator, and I generated pictures from the training set.
I got ~83000 pictures of signs including source training data, that is more or less distributed better than the original train data set

To generate pictures I used these parameters:
* zoom_range=0.2
* rotation_range=5
* preprocessing_function=image_preproc.imadjust
I added preprocessing function to make more suffisticated changes to the original picture.
So the train data set consist of ~35.000 untuched images and ~48.000 generated pictures

New signs distribution looks like this:

![visual](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/misc/augmentation1.png)

![visual](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/misc/augmentation2.png)

New distribution graph

![visual](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/misc/aug_training_set_distribution.png)

#### Result
[Link to the notebook](https://github.com/maslovw/SDND/blob/master/TrafficSignlassifier/Traffic_Sign_Classifier.ipynb)

`loss: 0.0149 - acc: 0.9953 - val_loss: 0.0571 - val_acc: 0.9848`

On the test data `loss: 0.07699867185758226, acc: 0.97846397466349966`

Which is better than previous, and has better speed perfomance agains my first approach

| model | val_loss| val_acc|
|:-----:|:-----:|:-----:|
|Multi-column			| 0.0883 | 0.9798|
|LeNet					| 0.2297 | 0.9617|
|LeNet on generated data| 0.0571 | 0.9848|

#### Layers

My final model consisted of the following layers:

| Layer         		|     Description	        					| Output Shape|
|:---------------------:|:---------------------------------------------:|:-------:|
| Input         		| RGB image   									| 32x32x3 |
| Lamda(x/127.5 - 1)	| RGB normalized image							| 32x32x3 |
| Convolution 3x3     	| 1x1 stride, same padding, 32 filters 			| 32x32x32|
| RELU					|												|         |
| Max pooling	      	| 2x2 stride									| 16x16x32|
| Convolution 3x3	    | 1x1 stride, valid padding, 64 filters			| 14x14x64|
| RELU					|												|         |
| Max pooling	      	| 2x2 stride									| 7x7x64  |
| Convolution 3x3	    | 1x1 stride, valid padding, 128 filters	    | 5x5x128 |
| RELU					|												|         |
| Flattern				|												| 3200    |
| Fully connected		| activation = 'relu'        					| 256     |
| Fully connected		|  activation = 'relu'							| 1024    |
| Softmax				|         										| 43      |


#### Training the model
I took Adam optimizer, it's recomended for CNN. 
To calculate gradient descent I used categorical crossentropy, again recommended method for classification tasks

To train the model, I used an batch size = 256.

I took 100 epochs, but training has EarlyStopping callback (if validation loss doesn't decreese in 20 epochs for more than 1e-03, then the training stops. 
I also used Learning rate scheduler: ReduceLROnPlateau. It reduces learning rate by 0.2 when a val_acc stopped improving in 3 epochs, minimum lr is 1e-05

To be able to take the best result, I included ModelCheckpoint callback, so  each epoch it saves the whole model.

##### My final model results were:
* training set accuracy of 0.9956
* validation set accuracy of 0.9850
* test set accuracy of 0.9785


### Test a Model on New Images

#### 1. Here are five German traffic signs that I found on the web:

![img2](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/gmaps/img_2.png)

![img1](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/gmaps/img_1.png)

![img8](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/gmaps/img_8.png)

![img9](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/gmaps/img_9.png)

![img4](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/gmaps/img_4.png)

The images with speed limit or digits on it are the most dificult for the model

#### 2. Results of the prediction:

|num| Image			        |     Prediction	|
|-|:---------------------:|:--------------------|
|1| SpeedLimit 20km/h	| Speed Limit 120km/h   |
|2| Slippery road     	| Slippery road 		|
|3| Speed limit (80km/h)| Speed limit (80km/h)	|
|4| Priority road		| Priority Road			|
|5| Keep right			| Keep right			|


The model was able to correctly guess 11 of the 12 traffic signs, which gives an accuracy of 91%. This compares favorably to the accuracy on the test set of 97%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the 3rd image, the model is relatively unsure that this is a Speed Limit 80 (probability of 0.45), and the image does contain a Speed limit 80. The top five soft max probabilities were
* (45.36%): Speed limit (80km/h)
* (36.13%): Speed limit (30km/h)
* (12.64%): End of speed limit (80km/h)
* (3.386%): Speed limit (50km/h)
* (1.027%): Speed limit (100km/h)

|num| Image			        |     Prediction	| Probability|
|-|:---------------------:|:--------------------:|:----:|
|1| SpeedLimit 20km/h	| Speed Limit 120km/h   | 99.91%|
|2| Slippery road     	| Slippery road 		| 100%  |
|3| Speed limit (80km/h)| Speed limit (80km/h)	| 45.36%|
|4| Priority road		| Priority Road			| 100%  |
|5| Keep right			| Keep right			| 100%  |


For the first image 
* (99.91%): Speed limit (120km/h)
* (0.07392%): Speed limit (20km/h)
* (0.01394%): Speed limit (70km/h)
* (0.001095%): Speed limit (100km/h)
* (0.0001956%): Speed limit (30km/h)
 
### Visualizing the Neural Network
Sorce image
![TurnRightAhead](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/misc/turn_right_ahead.png)

![Visualization](https://raw.githubusercontent.com/maslovw/SDND/master/TrafficSignlassifier/misc/visual_conv1.png)



