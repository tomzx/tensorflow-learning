import tensorflow as tf
import numpy as np

def main():
    layer_1_units = 128
    input_size = output_size = 512

    output_units = output_size

    inputs = tf.placeholder(tf.float32, [None, input_size])
    targets = tf.placeholder(tf.float32, [None, output_size])

    # Network definition
    layer = apply_dense(layer_1_units, inputs)
    layer = apply_dense(output_units, layer)

    outputs = layer

    # Loss definition
    loss = tf.reduce_mean(tf.square(outputs - targets))

    # Optimization
    optimizer = tf.train.AdamOptimizer()
    objective = optimizer.minimize(loss)

    # Prepare session feed_dict
    train_x, train_y = make_data(8000, output_units)
    test_x, test_y = make_data(2000, output_units)

    train_dictionary = {inputs: train_x, targets: train_y}
    test_dictionary = {inputs: test_x, targets: test_y}

    # Execute computation session
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for i in range(5000):
            if i % 100 == 0:
                print('epoch %d: loss=%f val_loss=%f' %
                      (i, session.run(loss, feed_dict=train_dictionary),
                       session.run(loss, feed_dict=test_dictionary)))
            session.run(objective, feed_dict=train_dictionary)

def apply_dense(units, inputs):
    weights = tf.Variable(tf.random_uniform((inputs.get_shape()[1].value, units)))
    biases = tf.Variable(tf.zeros((1, units)))

    return tf.matmul(inputs, weights) + biases

def make_data(samples, output_units):
    inputs = np.random.rand(samples, output_units)
    targets = inputs*inputs
    return inputs, targets

main()