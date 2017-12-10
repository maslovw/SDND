# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---
The goals/steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this GitHub repository:
* drive.py
* video.py
* writeup_template.md

## How it works

I tried many different CNN models (including  trained 
[ResNet50](https://github.com/maslovw/SDND/blob/master/BehavioralCloning/ResNet50.ipynb)).
But the best was the CNN introduced by NVIDIA ([link](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars))


[//]: # (Image References)
[model]: ./images/nvidia_model.png
[video]: ./images/VideoYouTube.png
[data_sample]: ./images/data_sample.png
[data_sample2]: ./images/data_sample2.png

### Dataset
I checked the dataset provided by Udacity, but I quickly realized that values of the steering wheel are
not normally distributed as I would expect to see. It has too many zeros inside. So I tried to record
my own data, changing the angle of a steering wheel with my mouse (I don't have joystick :( )

But at the end, I gave up and used initial Udacity dataset.
I used all three cameras making offset for side cameras - 0,2. I got 24,108 entries, which I also flipped 
vertically to make it harder for overfitting. 

Examples of data with used cropping:

![data_sample][data_sample]

![data_sample][data_sample2]

Note: I tried data augmentation and redistribution (can be observed in the BehaviorCloning.ipynb). 

### Model
On this data I tryed to train NN: 
```
input_5             : (160, 320, 1)
cropping2d_5        : (80, 320, 1)
lambda_5            : (80, 320, 1)
conv2d_24           : (80, 160, 32)
activation_19       : (80, 160, 32)
dropout_13          : (80, 160, 32)
max_pooling2d_19    : (80, 80, 32)
conv2d_25           : (80, 80, 48)
activation_20       : (80, 80, 48)
max_pooling2d_20    : (40, 40, 48)
conv2d_26           : (40, 40, 64)
activation_21       : (40, 40, 64)
dropout_14          : (40, 40, 64)
max_pooling2d_21    : (20, 20, 64)
conv2d_27           : (20, 20, 128)
activation_22       : (20, 20, 128)
max_pooling2d_22    : (10, 10, 128)
conv2d_28           : (8, 8, 256)
activation_23       : (8, 8, 256)
max_pooling2d_23    : (4, 4, 256)
flatten_5           : (4096,)
dense_17            : (256,)
dense_18            : (512,)
dense_19            : (1024,)
dropout_15          : (1024,)
dense_20            : (1,)
```

the best result was `loss: 0.1308 - val_loss: 0.0187` after 16 epoch and the car was making 
nearly successful attempts to stay on the track..

#### NVIDIA end-to-end
After many tryes, I decided to go with NVidia end-to-end approach, and I built this model:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
_________________________________________________________________
```

I used initial Udacity dataset.

And it worked! After only 2 epochs I got better loss results, and the car was actually driving 
through the track.

The model was trained with Adam optimizer. 
default Keras parameters: 

`keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)`

I tried SGD (Stochastic gradient descent optimizer), but adam showed a better result.
I tried to decrease learning rate depending on validation loss (in this notebook https://github.com/maslovw/SDND/blob/master/BehavioralCloning/BehaviorCloning.ipynb), but it didn't increase quality of training, 
in fact, it made it worse. 

Keras callback function 
`ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=2)` was used

To calculate loss I used MeanSquaredError(mse) function.

But I forgot to put activation function on the last Dense layers 

```
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
```
My colleague pointed on that, and said that if this works, then I should try to reduce
the Dense layers completely.

At the end, I've got the same result with model: 
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_4 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
lambda_4 (Lambda)            (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_17 (Conv2D)           (None, 20, 77, 36)        21636     
_________________________________________________________________
dropout_1 (Dropout)          (None, 20, 77, 36)        0         
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_20 (Conv2D)           (None, 4, 33, 64)         36928     
_________________________________________________________________
dropout_2 (Dropout)          (None, 4, 33, 64)         0         
_________________________________________________________________
flatten_4 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_10 (Dense)             (None, 1)                 8449      
=================================================================
Total params: 139,797
Trainable params: 139,797
Non-trainable params: 0
_________________________________________________________________

Train on 41465 samples, validate on 6751 samples
Epoch 1/3
158s - loss: 0.0198 - val_loss: 0.0186
Epoch 2/3
141s - loss: 0.0163 - val_loss: 0.0170
Epoch 3/3
139s - loss: 0.0153 - val_loss: 0.0169

```
Which is 7 times smaller than the dataset with Dense layers, and it behaves the same.

Here's the final notebook of training CNN (https://maslovw.github.io/SDND/BehaviorCloning/BehaviorCloning.html)

[![Video][video]](https://youtu.be/ZUaHttB-yYE)
Video https://youtu.be/ZUaHttB-yYE

the Model file is: https://github.com/maslovw/SDND/blob/master/BehavioralCloning/models/modelNv02.h5

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`.
See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py ./models/modelNv02.h5 imgs
```

The fourth argument, `imgs`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls imgs

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case, the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.
