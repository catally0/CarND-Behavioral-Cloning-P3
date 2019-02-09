# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./images/center.jpg "Cemter driving image"
[image4]: ./images/recovery.jpg "Recovery Image #1"
[image5]: ./images/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `train.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The data is normalized by a Lambda layer and cropped by a Cropping2D layer in Keras. (`train.py` lines 43-44)

After the data preprocessing, 5 convolution neural networks with 5x5 and 3x3 filter sizes and depths between 24 and 64 (`train.py` lines 41-49) are added. The activation function is RELU.

Then I added a flatten layer followed by 3 fully connected layers, whose outputs are from 100 to 10. Lastly the result, steering angle, is given by output layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (`train.py` lines 51). 

I put 20% random data from training data as the validation set to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`train.py` line 57).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The way I made the recovering data is that I firstly placed the vehicle in an dangerous position and heading, adjust the steering, then start recording and driving. When the vehicle returned to the center of the lane, I stopped recording. This is kind of tricky because I have to be very careful with the record timing. I did this several rounds, not only on the straight road, but also in the curve.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a picture from camera and the actual steering angle as inputs  and predict steering angle that can keep the vehicle running within the track.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it contains several convolution layers, which may help to extract the feature of the lane. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on both training set and validation set.  This implied that the model was underfitting. 

Then I added more Convolutional layers and more Dense layers. The MSE on both training and validation set declines.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially when the vehicle is about entering the curve. To improve the driving behavior in these cases, I drove the vehicle a couple of more times entering curves from different positions.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`train.py`, line 43 to 55) consisted of a convolution neural network with the following layers and layer sizes.

* Lambda layer: normalize the data
* Cropping layer: crop the image
* Convolution layer: Kernal size 5x5, depth 24, activation: relu
* Convolution layer: Kernal size 5x5, depth 36, activation: relu
* Convolution layer: Kernal size 5x5, depth 48, activation: relu
* Convolution layer: Kernal size 3x3, depth 64, activation: relu
* Convolution layer: Kernal size 3x3, depth 64, activation: relu
* Flatten layer
* Dense layer: output size: 100
* Dense layer: output size: 50
* Dense layer: output size: 10
* Output layer

#### 3. Creation of the Training Set & Training Process

As the simulator only provides a way to record the data, it is not easy to delete the data that we think they're not correct. For example, I may not be able to react correctly when the vehicle is running at a high speed. If this driving behavior is in the training dataset, the model will not perform well.

Considering this, I create the training data in several different smaller sets. Like 'Straight line driving', 'Recovering driving', 'Curve driving', etc. If I made a mistake in one of the driving cycle, I just scrap the data and start over again. This ensures the quality of the training data.

In order to make the driving more smooth, I used a Logitech G27 gaming steering wheel and convert the steering wheel axel input to mouse movement.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from bad positons or headings. These images show what a recovery looks like starting from the right side of a curve to the center of the road.

![alt text][image4]


I tried to drive on track #2. But it's too hard for me to stay on the track. I have to skip it. But driving track #1 clock-wise is also a good way, which helps to generalize the model.

To augment the data set, I also flipped images and angles thinking that this would help to balance the dataset. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

I also made use of the left/right camera image. I added an angle offset for them, namely add a right turn angle for the left image, and a left turn angle for the right image. So that the vehicle will tend to stay in the middle of the lane.

After the collection process, I had X number of data points. I then preprocessed this data by normalizing the pixel value. And cropped the image a bit to let the model focus on the road itself, instead of the trees, moutains, sky, etc.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. I put 20% random data as the validation set to help me determine if the model was over or under fitting. 

The ideal number of epochs was 2, because the loss starts to increasing on validation set, which is a sign of overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
