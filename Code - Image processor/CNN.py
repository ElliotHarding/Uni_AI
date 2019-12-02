import tensorflow as tf
from tensorflow.keras import datasets, models, backend, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

batch_size = 32
epochs = 100
IMG_HEIGHT = 150
IMG_WIDTH = 150

data_dir="aData\\"

classes = pd.read_csv("aData-names.csv")
class_names = classes.ClassName.tolist()




