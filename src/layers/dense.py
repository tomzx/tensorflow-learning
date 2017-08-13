import tensorflow as tf

from src import initializers
from src.layer import Layer
from src.layers import activations

class Dense(Layer):
    def __init__(self, units,
                 activation=None,
                 use_biases=True,
                 weights_initializer="random_normal",
                 biases_initializer="zeros",
                 write_histogram=True,
                 **kwargs):
        super(Dense, self).__init__(kwargs.get('input_shape'))
        self.write_histogram = write_histogram
        self.units = units
        self.activation = activations.get(activation)
        self.use_biases = use_biases
        self.weights_initializer = initializers.get(weights_initializer)
        self.biases_initializer = initializers.get(biases_initializer)

    def build(self, input_shape):
        weights_shape = (input_shape[1], self.units)
        initialized_weights = self.weights_initializer(weights_shape)
        self.weights = tf.Variable(initialized_weights, name='weights')
        if self.write_histogram:
            tf.summary.histogram('weights', self.weights)

        if self.use_biases:
            biases_shape = (1, self.units)
            initialized_biases = self.biases_initializer(biases_shape)
            self.biases = tf.Variable(initialized_biases, name='biases')
            if self.write_histogram:
                tf.summary.histogram('biases', self.biases)
        else:
            self.biases = None

    def call(self, inputs):
        output = tf.matmul(inputs, self.weights)

        if self.use_biases:
            output += self.biases

        if self.activation:
            output = self.activation(output)

        return output
