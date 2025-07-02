
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
import os

# Set image dimensions and paths
img_width, img_height = 150, 150
train_data_dir = "./training_set/training_set"
validation_data_dir = "./test_set/test_set"
batch_size = 20
epochs = 5

# Input shape based on image data format
input_shape = (img_width, img_height, 3)

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height),
                                                        batch_size=batch_size, class_mode='binary')

# Build CNN model
model = Sequential([
    Conv2D(64, (3, 3), input_shape=input_shape),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(64),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, steps_per_epoch=100, epochs=epochs,
          validation_data=validation_generator, validation_steps=100)

# Save model
model.save("dog_cat_classifier.h5")
