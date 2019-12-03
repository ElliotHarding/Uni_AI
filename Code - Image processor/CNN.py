import tensorflow as tf
from tensorflow.keras import datasets, models, backend, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#activate tf_gpu

batch_size = 12 # amount of images processed at once 8
epochs = 400
IMG_HEIGHT = 150
IMG_WIDTH = 150

data_dir = "C:\\aData"
classes = pd.read_csv('aData-names.csv')

data_dataGen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True) # Generator for our validation data

train_generator = data_dataGen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = data_dataGen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

sample_training_images, _ = next(train_generator)


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH , 3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(120, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Output summary of model
model.summary()

history = model.fit_generator(
    train_generator,
    steps_per_epoch= 6108 // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps= 2036 // batch_size
)

model.save('C:\\dog-cnn.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
