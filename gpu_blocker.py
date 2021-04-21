import tensorflow as tf
import time

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

with tf.Session() as sess:
    a=tf.constant(1.0)
    time.sleep(1000000)
        
