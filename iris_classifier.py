from sklearn import datasets

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

CHECKPOINT_FILE = './iris'

def main():
    iris_dataset = datasets.load_iris()
    # print(iris_dataset)

    iris_dataset.target = iris_dataset.target.reshape(-1, 1)
    encoder = OneHotEncoder()
    encoder.fit(iris_dataset.target)
    iris_dataset.target = encoder.transform(iris_dataset.target).toarray()

    output_size = iris_dataset.target.shape[1]

    inputs = tf.placeholder(tf.float32, [None, 4])
    targets = tf.placeholder(tf.float32, [None, output_size])

    layer = apply_dense(4, inputs)
    layer = tf.nn.relu(layer)
    layer = apply_dense(4, layer)
    layer = tf.nn.relu(layer)
    layer = apply_dense(output_size, layer)
    layer = tf.nn.softmax(layer)

    outputs = layer

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets))
    # loss = tf.reduce_mean(tf.square(outputs - targets))

    optimizer = tf.train.AdamOptimizer()
    objective = optimizer.minimize(loss)

    # Prepare data
    permutations = np.random.permutation(len(iris_dataset.data))
    data = iris_dataset.data[permutations]
    target = iris_dataset.target[permutations]

    train_length = int(len(data)*0.8)
    train_x = data[:train_length]
    train_y = target[:train_length]
    test_x = data[train_length:]
    test_y = target[train_length:]

    train_dictionary = {inputs: train_x, targets: train_y}
    test_dictionary = {inputs: test_x, targets: test_y}

    saver = tf.train.Saver()

    # Execute computation session
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        if os.path.exists(CHECKPOINT_FILE):
            saver.restore(session, CHECKPOINT_FILE)

        for i in range(10000):
            if i % 100 == 0:
                print('epoch %d: loss=%f val_loss=%f' %
                      (i, session.run(loss, feed_dict=train_dictionary),
                       session.run(loss, feed_dict=test_dictionary)))
                saver.save(session, CHECKPOINT_FILE)
            session.run(objective, feed_dict=train_dictionary)


        predictions = session.run(outputs, feed_dict=test_dictionary)

        out = list(zip(test_x, test_y, predictions))
        correct_count = 0
        count = len(predictions)
        for line in out:
            expected = np.argmax(line[1])
            predicted = np.argmax(line[2])
            correct = expected == predicted
            if correct:
                correct_count += 1
            print("{} {} {}".format(line[0], expected, predicted))
        print(correct_count/count)

def apply_dense(units, inputs):
    weights = tf.Variable(tf.random_uniform((inputs.get_shape()[1].value, units)))
    biases = tf.Variable(tf.zeros((1, units)))

    return tf.matmul(inputs, weights) + biases

main()