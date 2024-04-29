"""
keras_test.py

A simple script to test if keras has been installed properly. It builds a basic
feedforward network and displays a summary of the neural network architecture.

Keras is now part of Tensorflow, which is why you might see some warnings from the
Tensorflow core pop up as well.
"""
import keras


def run(print_output=False):
    input_layer = keras.layers.Input(shape=(784,))
    dense1 = keras.layers.Dense(64, activation='relu')(input_layer)
    output_layer = keras.layers.Dense(10, activation='softmax')(dense1)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    if print_output: model.summary()


if __name__ == "__main__":
    run(print_output=True)
