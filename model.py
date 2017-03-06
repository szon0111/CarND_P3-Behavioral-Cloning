import os
import csv

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

shuffle(lines)
train_data, validation_data = train_test_split(lines, test_size=0.2)

import numpy as np
import cv2

images = []
angles = []
for line in train_data:
    # use left, center, right camera images
    angle_correction = 0.1
    for i in range(3):
        name = './data/IMG/' + line[i].split('\\')[-1]
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        angle = float(line[3])
        # adjust steering wheel angle for different camera views
    angles.append(angle)
    angles.append(angle + angle_correction)
    angles.append(angle - angle_correction)
print("done loading {} images".format(len(images)))
images_with_flipped = []
angles_with_flipped = []
# add flipped images and angles
for image, angle in zip(images, angles):
    images_with_flipped.append(image)
    angles_with_flipped.append(angle)
    flipped_image = np.fliplr(image)
    flipped_angle = -angle
    images_with_flipped.append(flipped_image)
    angles_with_flipped.append(flipped_angle)
print("done flipping and loading {} images".format(len(images_with_flipped)))

# prepare validation set
val_images = []
val_angles = []
for line in validation_data:
    # use left, center, right camera images
    angle_correction = 0.1
    for i in range(3):
        name = './data/IMG/' + line[i].split('\\')[-1]
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        val_images.append(image)
        angle = float(line[3])
        # adjust steering wheel angle for different camera views
    val_angles.append(angle)
    val_angles.append(angle + angle_correction)
    val_angles.append(angle - angle_correction)
print("done loading {} validation images".format(len(val_images)))



X_train = np.array(images_with_flipped)
y_train = np.array(angles_with_flipped)
X_val = np.array(val_images)
y_val = np.array(val_angles)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data = (X_val, y_val), nb_epoch=3)

model.save('Lenet-model.h5')