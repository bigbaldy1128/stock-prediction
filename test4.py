from tensorflow.python import keras
import numpy as np
import tensorflow as tf

y_true = tf.Variable([1, 1, 0, 0, 1], dtype=tf.float32)
y_pred = tf.Variable([1, 0, 0, 0, 1], dtype=tf.float32)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(keras.metrics.binary_accuracy(y_true, y_pred)))
