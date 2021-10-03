# import the required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# Global Variables for later use
IMAGE_SIZE = 128
BATCH_SIZE = 32
CHANNELS = 3

# There are 26179 Images in our Dataset belonging to 10 different animals 
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "../input/animals10/raw-img",
    seed=123, # this will ensure we get the same images each time
    shuffle=True, # images inside the batches will be shuffled
    image_size=(IMAGE_SIZE,IMAGE_SIZE), # every image will be of 128x128 dimention
    batch_size=BATCH_SIZE # There will be 32 images in each batch
)

# Class names are attached as a seperate text file
class_names = dataset.class_names

# split the dataset in train, test, val sets
# total batches of data = 819
train_ds = dataset.take(655) # 80% of 819
test_ds = dataset.skip(655) # remaining 20%
val_ds = test_ds.take(82) # 10% of 819
test_ds = test_ds.skip(82) # 10% of 819

# Data Augmentation Layer 
# Augmentation is the process of creating new training samples by altering the available data
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

# Applying Augmentation on Training Data
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

# Designing and Training the Model

# Reshaping so that each image is of same size and rescaling images them for normalization
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1./255),
])

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 90

# creating a sequential model
model = tf.keras.Sequential([
    resize_and_rescale,
    Conv2D(16, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(32,  kernel_size = (3,3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(32,  kernel_size = (3,3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)

# Slowing down the learning rate
opt = optimizers.Adam(learning_rate=0.0001)

# compile the model
model.compile(loss = 'sparse_categorical_crossentropy', optimizer=opt, metrics= ["accuracy"])

# use early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# save the best model with least validation loss
checkpointer = ModelCheckpoint(filepath="animal_weights.h5", verbose=1, save_best_only=True)

# Train the model
history = model.fit(train_ds, epochs = 100, validation_data=val_ds, batch_size=BATCH_SIZE, shuffle=True, callbacks=[earlystopping, checkpointer])

# save the model architecture to json file for future use
model_json = model.to_json()
with open(BASE_DIR/"animal_model.json","w") as json_file:
    json_file.write(model_json)

# Load pretrained model (best saved one)
with open(BASE_DIR/'animal_model.json', 'r') as json_file:
    json_savedModel= json_file.read()
# load the model  
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights(BASE_DIR/'animal_weights.h5')
model.compile(loss = 'sparse_categorical_crossentropy', optimizer=opt, metrics= ["accuracy"])

scores = model.evaluate(test_ds)

print("Accuracy on test set = ", scores[1])