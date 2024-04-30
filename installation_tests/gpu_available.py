"""
gpu_available.py

Using the GPU for computation is a lot faster. Let's see if it is available.

Not recognizing GPU? The following article has a guide (mostly NVidia)
https://saturncloud.io/blog/what-to-do-when-tensorflow-is-not-detecting-your-gpu/
"""
import tensorflow as tf

# Check available devices
devices = tf.config.list_physical_devices()
print("Available devices:", devices)

# Check if GPU is available
gpu_available = tf.config.list_physical_devices('GPU')
if gpu_available:
    print("GPU is available")
else:
    print("GPU is not available")
