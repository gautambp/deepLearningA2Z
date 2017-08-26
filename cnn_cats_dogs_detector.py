# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# Part 1 - Build CNN Classifier
#initialize classifier
classifier = Sequential()

# Step 1 - Add convolution layer
# filters/features count = 32
# size of each filter extractor is 3x3
# input file size - 64x64 pixels with 3 colors for each pixel
# activation function = relu (rectifier activation func)
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), input_shape=(64,64,3), activation='relu'))

# Step 2 - Pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flatten the previous layer of pooling
classifier.add(Flatten())

# Step 4 - Add hidden layer with 128 output nodes
# Input connection implied from previous layers (64x64 * 32 / 2)
classifier.add(Dense(units = 128, activation='relu'))

# Step 5 - Add output layer
# One output and the use of sigmoid activation function
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile and classifier..
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Train the model with cat/dog images training set

# Step 1 - Perform image augmentation to increase train/test size
# use various image processing techniques to generate more images (zooming, shearing, flipping)
# from existing images set

from keras.preprocessing.image import ImageDataGenerator

# scale color to 1..255
# shear range is 0.2 .. zoom is also 0.2.. perform horz flip
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# scale the images to 64x64
train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# we've 8000 images in train set and 2000 images in test set
classifier.fit_generator(
        train_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
