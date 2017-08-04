import numpy as np

from src.layers.activation import Activation
from src.layers.dense import Dense
from src.layers.dropout import Dropout
from src.layers.input import Input
from src.model import Model

def main():
    input_size = output_size = 1024

    inputs = Input((input_size,))
    layer = Dense(5)(inputs)
    layer = Activation('relu')(layer)
    layer = Dropout(0.25)(layer)
    outputs = Dense(output_size)(layer)

    train_x, train_y = make_data(80000, output_size)
    test_x, test_y = make_data(20000, output_size)

    model = Model(inputs, outputs)
    model.compile('adam', 'mean_squared_error')
    model.fit(train_x, train_y, epochs=20, validation_data=(test_x, test_y))

def make_data(samples, output_units):
    inputs = np.random.rand(samples, output_units)
    targets = inputs * inputs
    return inputs, targets

main()
