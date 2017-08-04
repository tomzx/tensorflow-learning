import tensorflow as tf

from src.layer import Layer

class Dense(Layer):
    def __init__(self, units, **kwargs):
        super(Dense, self).__init__(kwargs.get('input_shape'))
        self.units = units

    def build(self, input_shape):
        self.weights = tf.Variable(tf.random_uniform((input_shape[1], self.units)))
        self.biases = tf.Variable(tf.zeros((1, self.units)))

    def call(self, inputs):
        return tf.matmul(inputs, self.weights) + self.biases
