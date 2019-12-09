#%tensorflow_version 2.4

#Imports
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


#Settings
image_size=244
batch_size=32
epochs=250


#Get pre-traied model & apply layers
base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #add dense layers so that model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x)
x=Dense(512,activation='relu')(x) 
preds=Dense(120,activation='softmax')(x)#final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True



# Prep data
data_dir = "C://aData"

data_dataGen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2, horizontal_flip=True) # Generator for our validation data

train_data = data_dataGen.flow_from_directory(
    data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    subset='training',
    shuffle=True) # set as training data

validation_data = data_dataGen.flow_from_directory(
    data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation') # set as validation data


#Compile model
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='E://dog-cnn-savepoint.h5',
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        save_best_only=True,
        monitor='val_loss',
        verbose=1)#,
    #keras.callbacks.EarlyStopping(monitor='val_loss')
]

#Run on GPU
with tf.device("/gpu:0"):
  history = model.fit_generator(generator=train_data,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    steps_per_epoch=train_data.samples/batch_size,
                    epochs=epochs)

#Save model
model.save('E://dog-cnnn.h5')




#Output results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

#epochs_range = range(epochs)

#plt.figure(figsize=(8, 8))
#plt.subplot(1, 2, 1)
#plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.title('Training and Validation Accuracy')

#plt.subplot(1, 2, 2)
#plt.plot(epochs_range, loss, label='Training Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.title('Training and Validation Loss')
#plt.show()
