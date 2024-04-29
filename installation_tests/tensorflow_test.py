"""
tensorflow_test.py

Run this little script to see if tensorflow is installed properly.
You might get a couple of warnings and configuration tips, but the
program should end with the following line in your terminal:

`Sum of 3.0 and 4.0 is 7.0`

After printing, the program should exit successfully.
"""

import tensorflow as tf


def run(print_output=False):
    a = tf.constant(3.0)
    b = tf.constant(4.0)
    result = a + b

    if print_output: print(f"Sum of {a.numpy()} and {b.numpy()} is {result.numpy()}")


if __name__ == "__main__":
    run(print_output=True)
