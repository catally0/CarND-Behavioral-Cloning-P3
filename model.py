import csv
import cv2
import numpy as np

# Load training datasets
lines=[]
# I record small datasets in simulator and combine them in a big one, see detail in report
datasets=['general_center_driving','curve_driving','recovery_driving']
for dataset in datasets:
    print('Start loading training dataset: %s' % dataset)
    with open('../driving_data/'+dataset+'/driving_log.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line[0]= '../driving_data/'+dataset+'/IMG/'+line[0].split('\\')[-1]
            line[1]= '../driving_data/'+dataset+'/IMG/'+line[1].split('\\')[-1]
            line[2]= '../driving_data/'+dataset+'/IMG/'+line[2].split('\\')[-1]
            lines.append(line)

images=[]
angles=[]
for line in lines:
    #print(line)
    angle_correction = [0.0, 0.3, -0.3] #Augment the data by using left/right camera and flip the image.
    for i in range(3):
        img=cv2.imread(line[i])
        angle=float(line[3]) + angle_correction[i]
        images.append(img)
        angles.append(angle)
        img_flipped = np.fliplr(img)
        angle_flipped = -angle
        images.append(img_flipped)
        angles.append(angle_flipped)  

images = np.array(images)
angles = np.array(angles)

X_train = images
y_train = angles

# Load validation datasets
lines=[]
datasets=['general_center_driving']
for dataset in datasets:
    print('Start loading validation dataset: %s' % dataset)
    with open('../driving_data/'+dataset+'/driving_log.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line[0]= '../driving_data/'+dataset+'/IMG/'+line[0].split('\\')[-1]
            line[1]= '../driving_data/'+dataset+'/IMG/'+line[1].split('\\')[-1]
            line[2]= '../driving_data/'+dataset+'/IMG/'+line[2].split('\\')[-1]
            lines.append(line)

images=[]
angles=[]
for line in lines:
    #print(line)
    angle_correction = [0.0, 0.3, -0.3] #Augment the data by using left/right camera and flip the image.
    for i in range(1):
        img=cv2.imread(line[i])
        angle=float(line[3]) + angle_correction[i]
        images.append(img)
        angles.append(angle)
        img_flipped = np.fliplr(img)
        angle_flipped = -angle
        images.append(img_flipped)
        angles.append(angle_flipped)  

images = np.array(images)
angles = np.array(angles)

X_val = images
y_val = angles

# Build the model
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint

model=Sequential()
model.add(Lambda(lambda x:(x/255) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20), (0,0))))
model.add(Conv2D(24,(5,5),strides=(2, 2),activation='relu'))
model.add(Conv2D(36,(5,5),strides=(2, 2),activation='relu'))
model.add(Conv2D(48,(5,5),strides=(2, 2),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
#model.add(Dropout(0.1))
model.add(Dense(100,activation='relu'))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


# Define optimzer and train the model
model.compile(loss='mse', optimizer='adam')

checkpoint = ModelCheckpoint('./model1/{epoch:02d}_{val_loss:.4f}.h5', verbose=True, monitor='val_loss',save_best_only=False, mode='auto') 
model.fit(X_train, y_train, validation_data=(X_val, y_val), shuffle=True, epochs=5, callbacks=[checkpoint])
