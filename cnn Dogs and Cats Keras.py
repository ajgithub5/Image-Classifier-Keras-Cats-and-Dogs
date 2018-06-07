#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:57:17 2018

@author: ajay
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Building CNN network

#Step 1 import all the necessary keras package required to make CNN model

from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Conv2D

#Initialisation the CNN
classifier = Sequential()

#classifier.add(Convolution2D(32,3,3, border_mode = 'same', input_shape=(64,64,3),activation='relu'))
classifier.add(Conv2D(32,(3,3),activation="relu",input_shape = (64,64,3)))

#Maxpooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a second a convolutional layer
classifier.add(Conv2D(32,(3,3),activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a third convolutional layer
classifier.add(Conv2D(32,(3,3),activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flattening
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(activation='relu',output_dim=128))
classifier.add(Dense(output_dim =1,activation='sigmoid'))

# Compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics =['accuracy'])

# fitting the cnn to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

trainpath = '/home/ajay/AJAY/My working Files/Udemy Jupyter/Convolutional_Neural_Networks/original_data_cd/train_set'
testpath = '/home/ajay/AJAY/My working Files/Udemy Jupyter/Convolutional_Neural_Networks/original_data_cd/test_set'

training_set = train_datagen.flow_from_directory(trainpath,
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(testpath,
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    samples_per_epoch=8000,
                    epochs=10,
                    validation_data=test_set,
                    nb_val_samples=2000)

# predicting new image

import numpy as np
from keras.preprocessing import image
def predict_image(imgpath):
    test_image = image.load_img(imgpath,target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    result=classifier.predict(test_image)
    training_set.class_indices
    if result[0][0]==1:
        prediction = "Dog"
    else:
        prediction = "Cat"
    print(prediction)
    
imgpath = '/home/ajay/AJAY/My working Files/Udemy Jupyter/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_1.jpg'
predict_image(imgpath)


#test_image1 = image.load_img('/home/ajay/AJAY/My working Files/Udemy Jupyter/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
#test_image1 = image.img_to_array(test_image1)
#test_image = np.expand_dims(test_image1,axis = 0)
#result = training_set.class_indices








