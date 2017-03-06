# Project 3: Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
Train, validate and test a model using Keras to clone driving behavior. The model will output a steering angle to an autonomous vehicle.
Data will be collected by running a car manually in Udacity's Self-Driving Car simulator. The model performance will be tested by running the car in the simulator in autonomous mode.

####The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Project Deliverables
---
* `model.py` contains the code for building the keras model based on the [Nvidia architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). 
* `Nvidia-model-epoch5.h5` is the trained keras model needed to run the vehicle autonomously.
* `drive.py` is used to drive the vehicle autonomously in the simulator. It accepts an h5 file as a parameter.

Data Collection
---
Image data and corresponding steering angle were collected by manually driving the vehicle in the simulator for 3 laps. The vehicle has 3 cameras to record the images - left, center, and right. In total, 10,446 (3482 x 3) images were collected.
The steering angles for the images from the left camera were adjusted by 0.1, while the angles for the images from the right camera were adjusted by -0.1 to consider different perspective view from each camera.

Data Augmentation
---
Horizontally flipped images and opposite steering angles were added as the vehicle runs counterclockwise in the training track, causing left turn bias.

Data Pre-processing
---
Input images are normalized and cropped using the keras functions to take advantage of the power of GPU. The top 70 pixels that include features like sky, trees, etc and the bottom 20 pixels that show the hood of the vehicle are not needed and thus removed - this can also help reduce the computing power needed.

Model Architecture
---
After trying various models including the good ol' LeNet-5, I settled with the [Nvidia architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). It includes a normalization layer, 5 convolutional layers, and 3 fully connected layers.

Results
---
View the [video](https://youtu.be/jL5L7-OCQxE) on youtube
