import datetime

import numpy as np

from src.callbacks import TensorBoard
from src.layers.dense import Dense
from src.layers.dropout import Dropout
from src.layers.input import Input
from src.model import Model

def main():
    input_size = output_size = 2048

    now = datetime.datetime.now()
    time = now.strftime('%Y.%m.%d %H.%M')

    inputs = Input((input_size,))
    layer = Dense(5, activation='relu')(inputs)
    layer = Dropout(0.25)(layer)
    outputs = Dense(output_size, activation='relu')(layer)

    train_x, train_y = make_data(80000, output_size)
    test_x, test_y = make_data(20000, output_size)

    tensorboard = TensorBoard('./logs/' + time)

    model = Model(inputs, outputs)
    model.compile('adam', 'mean_squared_error')
    model.fit(train_x, train_y,
              epochs=20,
              validation_data=(test_x, test_y),
              callbacks=[tensorboard]
              )

def make_data(samples, output_units):
    inputs = np.random.rand(samples, output_units)
    targets = inputs * inputs
    return inputs, targets

main()
