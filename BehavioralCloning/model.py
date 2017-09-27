# # Training CNN

import numpy as np
import cv2
from sklearn.utils import shuffle
from scipy.misc import imresize
import pandas as pd
import os

from keras.models import Model
from keras.layers import Cropping2D, Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers import Lambda, concatenate, Activation
from keras import initializers
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential

import pickle

# ## Load dataset

data_path = './data/data/'

path = data_path+'driving_log.csv'
log = pd.read_csv(path)


def read_line(path, line, camera, offset=0):
    """
    reads image from line[camera]
    returns steering angele with offset and image
    """
    ipath = line[camera].strip()
    path = os.path.join(path, ipath)
    #print(path)
    image = cv2.imread(path)
    if image is None:
        print('cant open an image', path)
        return (None, None)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    steering = float(line['steering'])
    if abs(steering) < (1 - abs(offset)):
        steering += offset
    return steering, image_rgb

strn, imgrgb = read_line(data_path, log.iloc[4000], 'center')

images_rgb = []
steerings = []

def readAllImgs(logs, path):
    def readAndAppend(path, en, camera, offset):
        steering, rgb = read_line(path, en, camera, offset)
        if rgm is None:
            return False
        steerings.append(steering)
        images_rgb.append(rgb)
        return True
    
    for i,en in log.iterrows():
        ret = readAndAppend(path, en, 'center', 0)
        if not ret:
            continue
        readAndAppend(path, en, 'left', 0.2)
        readAndAppend(path, en, 'right', -0.2)

    
# load all the images and steering data
readAllImgs(log, data_path)

# ## Building model
def buildModel():
    model = Sequential()
    model = Sequential()
    # CNN (ref: NVIDIA End-to-end model)
    # crop road part
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    # normalize data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.7))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Dropout(0.7))
    model.add(Flatten())
    model.add(Dense(1))
    return model

model = buildModel()

print(model.summary())

model.compile('adam', 'mse')

# ### Prepare data
XA = np.array(images_rgb)
YA = np.array(steerings)

# #### Append symmetrical data
# ##### Flip all the pictures vertically
XB = np.flip(XA, axis=2)
YB = YA * (-1)

# #### Concatenate arrays together
X = np.concatenate((XA, XB))
Y = np.concatenate((YA, YB))

# free memory
del XA
del XB
del data

X, Y = shuffle((X,Y))

print('X.shape', X.shape)
# ### Train the model
model.fit(X,Y, batch_size=128, epochs=2, verbose=2, validation_split=0.14)

model.save('models/modelNv02.h5')

print('Done')

