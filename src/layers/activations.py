import tensorflow as tf

from src.layer import Layer

class Activation(Layer):
    def __init__(self, name):
        super(Activation, self).__init__()
        self.name = name

    def call(self, inputs):
        if self.name == 'relu':
            return relu(inputs)
        elif self.name == 'sigmoid':
            return sigmoid(inputs)
        elif self.name == 'elu':
            return elu(inputs)
        elif self.name == 'tanh':
            return tanh(inputs)
        else:
            raise ValueError('Invalid activation type {}' % self.name)


def relu(inputs):
    return tf.nn.relu(inputs)

def sigmoid(inputs):
    return tf.nn.sigmoid(inputs)

def elu(inputs):
    return tf.nn.elu(inputs)

def tanh(inputs):
    return tf.nn.tanh(inputs)

def linear(inputs):
    return inputs

def get(identifier):
    module_globals = globals()
    if identifier is None:
        return linear
    elif identifier in module_globals and callable(module_globals[identifier]):
        return module_globals[identifier]
    else:
        raise ValueError('Invalid activation type {}' % identifier)