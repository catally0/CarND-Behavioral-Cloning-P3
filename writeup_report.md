# **Behavioral Cloning** 


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
* `train-model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `./model1/*.h5` containing a trained convolution neural network 
* `video-model1-run?.mp4` is the video of autonomous driving by different models
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py ./model1/{filename}.h5
```

#### 3. Submission code is usable and readable

The `train.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The data is normalized by a Lambda layer and cropped by a Cropping2D layer in Keras.

After the data preprocessing, 5 convolution neural networks with 5x5 and 3x3 filter sizes and depths between 24 and 64 are added. The activation function is RELU.

Then I added a flatten layer followed by 3 fully connected layers, whose outputs are from 100 to 10. Lastly the result, steering angle, is given by output layer.

#### 2. Attempts to reduce overfitting in the model

I tried to addd some dropout layers and take some samples from track #2 for training, in order to reduce overfitting. The vehicle can drive on some part of new tracks. But it turned out that the the vehicle will drive out of the track more frequently than before.

The attempt to reduce overfitting makes me to reflect on the whole goal of project. The target is to make the vehicle to stay on the track #1. It means even if the model runs out of track only in one single scenario, the project fails.

So after a couple of trails, I gave up trying to reduce overfitting. I just want to make sure the vehicle can run smoothly on track #1.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of general center lane driving, central curve drving and recovering driving from the left and right sides of the road. The way I made the recovering data is that I firstly placed the vehicle in an dangerous position and heading, adjust the steering, then start recording and driving. When the vehicle returned to the center of the lane, I stopped recording. This is kind of tricky because I have to be very careful with the record timing. I did this several rounds, not only on the straight road, but also in the curve.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a picture from camera and the actual steering angle as inputs  and predict steering angle that can keep the vehicle running within the track.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it contains several convolution layers, which may help to extract the feature of the lane. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on both training set and validation set.  This implied that the model was underfitting. 

Then I added more Convolutional layers and more Dense layers. The MSE on both training and validation set declines.

I also tried to add dropout layer, trying to reduce overfitting. But it turned out bad. The vehicle runs out of track more frequently than before.

Later I realized that a low MSE can reflect the performance of the model to an extend. But it doesn't guarantee the vehicle will stay on the track. Sometimes even if the model gives bad values for very small part of images, the vehicle will run out of track. Considering this, I cannot rely on the loss value, but to check the result manully for all epochs.

For the last step, I run the simulator for all the epochs after training a model, to see how well the car was driving around track one. If none of the model works, I will adjust the parameter of layers, including Conv, Dense layers, as well as the training datasets.

In the end, I pick the best performing models and generates the video. All the models that work are saved in `./model`.

#### 2. Final Model Architecture

The model architecture for model1 (train-model1.py) consisted of a convolution neural network with the following layers and layer sizes.

This architecture refers to https://devblogs.nvidia.com/deep-learning-self-driving-cars/

* Lambda layer: normalize the data
* Cropping layer: crop the image
* Convolution layer: Kernal size 5x5, depth 24, strides 2x2, activation: relu
* Convolution layer: Kernal size 5x5, depth 36, strides 2x2, activation: relu
* Convolution layer: Kernal size 5x5, depth 48, strides 2x2, activation: relu
* Convolution layer: Kernal size 3x3, depth 64, activation: relu
* Convolution layer: Kernal size 3x3, depth 64, activation: relu
* Flatten layer
* Dense layer: output size: 100
* Dense layer: output size: 50
* Dense layer: output size: 10
* Output layer

#### 3. Creation of the Training Set & Training Process

As the simulator only provides a way to record the data, it is not easy to delete the data that we think they're not correct. For example, I may not be able to react correctly when the vehicle is running at a high speed. If bad driving behavior is included in the training dataset, it is expected that the model will not perform well.

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

After the collection process, I had over 50k images. I then preprocessed this data by normalizing the pixel value. And cropped the image a bit to let the model focus on the road itself, instead of the trees, moutains, sky, etc.

I used all data for training the model. For validation data, I use only the general center driving, center camera data.

I have to manually simulate all the models generated from the each training epoch. As the lowest MSE doesn't mean it can drive within the track. But basically in 5 epochs, the model will have the best performance. I put an `[OK]` label in front of the valid model name and save them in `./model`.