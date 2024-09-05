import tensorflow as tf
import datetime
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Load and preprocess data
class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self):
        images, labels = [], []
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                label = int(class_dir)
                for image_file in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_file)
                    image = load_img(image_path, target_size=(32, 32))
                    image = img_to_array(image)
                    images.append(image)
                    labels.append(label)

        images = np.array(images, dtype='float32') / 255.0
        labels = np.array(labels)
        images, labels = shuffle(images, labels)
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        return x_train, y_train, x_test, y_test

# Model definition and training
class TrafficSignModel:
    def __init__(self, input_shape=(32, 32, 3), num_classes=43):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        # Define the feature extractor part (convolutional layers)
        model = Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        return model

    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        # Compile the model with optimizer, loss, and metrics
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model

# Trainer class for managing the training process
class Trainer:
    def __init__(self, data_dir='data/GTSRB/Final_Training/Images'):
        self.data_dir = data_dir

    def train(self, optimizer='adam', loss='sparse_categorical_crossentropy', epochs=10):
        loader = DataLoader(self.data_dir)
        x_train, y_train, x_test, y_test = loader.load_data()

        model_builder = TrafficSignModel(input_shape=(32, 32, 3), num_classes=43)
        model = model_builder.create_model()
        model = model_builder.compile_model(optimizer=optimizer, loss=loss)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
        model.save(f'model/Traffic_detection_{optimizer}_{loss}.h5')

        return model
