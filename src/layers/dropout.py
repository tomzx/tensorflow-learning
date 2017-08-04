import tensorflow as tf

from src.layer import Layer

class Dropout(Layer):
    def __init__(self, drop_probability):
        super(Dropout, self).__init__()
        self.drop_probability = drop_probability

    def call(self, inputs):
        return tf.nn.dropout(inputs, 1 - self.drop_probability)