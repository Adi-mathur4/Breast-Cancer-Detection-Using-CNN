# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:57:03 2020

@author: adity
"""

from keras.models import Sequential 
from keras.layers import Dropout,Dense,Flatten,MaxPooling2D,Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization

model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = (700,460,3),activation = 'relu'))
#model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
#model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation='relu'))
#model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(256,(3,3),activation='relu'))
#model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(512,(3,3),activation='relu'))
#model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
#model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1,activation='relu'))
#model.add(Activation('relu'))
model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('D:/CNN/fold1/train',target_size = (700, 460),batch_size = 32,class_mode = 'binary')
test_set = test_datagen.flow_from_directory('D:/CNN/fold1/test',target_size = (700, 460),batch_size = 32,class_mode = 'binary')
model.fit_generator(training_set,steps_per_epoch = 5005, epochs = 3, validation_data = test_set, validation_steps = 2904)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('D:/CNN/c.png', target_size = (700, 460))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Benign'
    print('Benign')
else:
    prediction = 'malignant'
    print('Malignant')