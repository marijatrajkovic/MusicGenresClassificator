from settings import *
from data import *
import tensorflow as tf
import keras
import numpy as np

def create_model(x_train, x_test, classes, trainable_encoder=False):
    # input shape (128, 2584, 1)

    model = keras.models.Sequential([
        keras.layers.InputLayer((128, 2584)),
        keras.layers.Reshape((128, 2584, 1)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.GlobalAvgPool2D(),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy])

    model.summary()

    return model