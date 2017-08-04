import tensorflow as tf

from src.layer import Layer

class Activation(Layer):
    def __init__(self, name):
        super(Activation, self).__init__()
        self.name = name

    def call(self, inputs):
        if self.name == 'relu':
            return tf.nn.relu(inputs)
        elif self.name == 'sigmoid':
            return tf.nn.sigmoid(inputs)
        elif self.name == 'elu':
            return tf.nn.elu(inputs)
        elif self.name == 'tanh':
            return tf.nn.tanh(inputs)
        else:
            raise ValueError('Invalid activation type')