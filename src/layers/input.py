import tensorflow as tf

def Input(input_shape):
    return tf.placeholder(tf.float32, [None] + list(input_shape))
