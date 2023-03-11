# Import the required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Download the data and set key variables
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
    -O /tmp/cats_and_dogs_filtered.zip
zip_ref = zipfile.ZipFile('/tmp/cats_and_dogs_filtered.zip', 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

train_dir = '/tmp/cats_and_dogs_filtered/train'
validation_dir = '/tmp/cats_and_dogs_filtered/validation'

# Define the image size and batch size
IMG_SIZE = 150
BATCH_SIZE = 32

# Create an ImageDataGenerator for the training data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Create an ImageDataGenerator for the validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# Use the ImageDataGenerators to load the data from the directories
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(IMG_SIZE, IMG_SIZE),
                                               batch_size=BATCH_SIZE,
                                               class_mode='binary')

validation_data = validation_datagen.flow_from_directory(validation_dir,
                                                         target_size=(IMG_SIZE, IMG_SIZE),
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='binary')

# Define the CNN model
model = tf.keras.models.Sequential([
    # Add the input layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    # Add another convolutional layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Add a third convolutional layer
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the output from the convolutional layers
    tf.keras.layers.Flatten(),
    # Add a fully connected layer with 512 neurons and a relu activation function
    tf.keras.layers.Dense(512, activation='relu'),
    # Add a dropout layer to reduce overfitting
    tf.keras.layers.Dropout(0.5),
    # Add the output layer with a single neuron and a sigmoid activation function
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data,
                    epochs=50,
                    steps_per_epoch=train_data.samples//BATCH_SIZE,
                    validation_data=validation_data,
                    validation_steps=validation_data.samples//BATCH_SIZE)

# Plot the training and validation accuracy over time
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
