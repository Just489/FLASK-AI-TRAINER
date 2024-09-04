import tensorflow as tf
from tensorflow.keras import layers, models

class TrafficSignModel:
    def __init__(self, input_shape=(32, 32, 3), num_classes=43):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        # Define the feature extractor part (convolutional layers)
        model = models.Sequential([
            # First convolutional block with 32 filters and a 3x3 kernel, using ReLU activation
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second convolutional block with 64 filters
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third convolutional block with 128 filters
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Flattening the convolutions for fully connected layers
            layers.Flatten()
        ])

        # Add classifier layers (fully connected dense layers)
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        return model

    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        # Compile the model with optimizer, loss, and metrics
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Example of how to instantiate and use the model
traffic_sign_model = TrafficSignModel()
traffic_sign_model.compile_model()