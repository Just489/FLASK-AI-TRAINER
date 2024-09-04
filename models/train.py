import tensorflow as tf
import datetime
from ts_model import TrafficSignModel
from ts_model import parameter, model
import numpy as np
import os
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image  import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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

class Trainer:
    def __init__(self, data_dir='data/GTSRB/Final_Training/Images'):
        self.data_dir = data_dir

    def train(self, optimizer='adam', loss='sparse_categorical_crossentropy', epochs=10):
        loader = DataLoader(self.data_dir)
        x_train, y_train, x_test, y_test = loader.load_data()

        model_builder = TrafficSignModel(input_shape=(32, 32, 3), num_classes=43)
        model = model_builder.create_model()
        model = model_builder.compile_model(model, optimizer=optimizer, loss=loss)

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
        model.save(f'model/Traffic_detection_{optimizer}_{loss}.h5')

        return model
