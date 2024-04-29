"""
keras_test.py

A simple script to test if keras has been installed properly. It builds a basic
feedforward network and displays a summary of the neural network architecture.

Keras is now part of Tensorflow, which is why you might see some
"""
import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

model.summary()
