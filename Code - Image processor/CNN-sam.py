import tensorflow as tf
from tensorflow.keras import datasets, models, backend, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

batch_size = 32
epochs = 100
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_dir = "C:\\Users\\elliot\\Desktop\\test split into two\\train"
test_dir = "C:\\Users\\elliot\\Desktop\\test split into two\\test"

classes = pd.read_csv("classnames.csv")
class_names = classes.ClassName.tolist()

train_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=30, zoom_range=0.5) # Generator for our training data
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')


#Show 5 augmented styles
#augmented_images = [train_data_gen[0][0][0] for i in range(5)]
#plotImages(augmented_images)

validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

test_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)

#Show 5 sample images
#plotImages(sample_training_images[:5])

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
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Output summary of model
model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch= 6108 // batch_size,
    epochs=epochs,
    validation_data=test_data_gen,
    validation_steps= 2036 // batch_size
)

model.save('convolutional.h5')

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
########################################################
#Manualy scale image data to 1, from grey scale image
#THIS WILL NEED TO BE CHANGED, AS INSTEAD OF ONE LAYER OF GREY SCALE, IT WILL BE 3 LAYERS OF RGB
# train_images = train_images / 255
# test_images = test_images / 255

# #Force fourth dimension for the CN, force with adding 1
# img_rows = train_images.shape[1]
# img_cols = train_images.shape[2]
# if backend.image_data_format() == "channels_first":
#     train_images = train_images.reshape(train_images.shape[0], 1, img_rows, img_cols)
#     test_images = test_images.reshape(test_images.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
#     test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# #Create CNN
# model = models.Sequential() #Goes through layers sequentially
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape)) #Conveusion 2D Layer
# model.add(layers.MaxPooling2D((2,2))) #Step down amount of neurons
# model.add(layers.Conv2D(64, (3,3), activation='relu')) #Conveusion 2D Layer
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu')) #Conveusion 2D Layer
# #Flatten and add Dense Layers
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
# #Display
# model.summary()

# #Compile model
# model.compile(optimize='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# #Train model
# #Train and validate 10 times
# history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# #Evaluate
# test_loss, test_acc, = model.evaluate(test_images, test_labels, verbose=2)

# print("\nTest Accuracy: ", test_acc)
# model.save('convolutional.h5')