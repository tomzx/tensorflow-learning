import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

class Model:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.compiled = False

        self.targets = tf.placeholder(tf.float32, outputs.get_shape())

    def fit(self, x, y, batch_size=32, epochs=10, verbose=1, validation_data=None, shuffle=True):
        if not self.compiled:
            raise RuntimeError('Model has not been compiled!')

        if x.shape[0] != y.shape[0]:
            raise RuntimeError('x and y are not of the same length')

        summaries = tf.summary.merge_all()
        now = datetime.datetime.now()
        time = now.strftime('%Y.%m.%d %H.%M')
        writer = tf.summary.FileWriter('./logs/' + time)

        # Execute computation session
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            writer.add_graph(session.graph)

            for epoch in tqdm(range(1, epochs+1), desc='Epoch', unit='epoch'):
                if shuffle:
                    permutations = np.random.permutation(x.shape[0])
                    x = x[permutations]
                    y = y[permutations]

                for batch in tqdm(range(0, x.shape[0], batch_size), desc='Batch', unit='sample', unit_scale=batch_size):
                    # Prepare feed dictionaries given the batch size
                    train_dictionary = {self.inputs: x[batch:batch+batch_size], self.targets: y[batch:batch+batch_size]}

                    session.run(self.objective, feed_dict=train_dictionary)

                test_dictionary = {self.inputs: validation_data[0],
                                   self.targets: validation_data[1]} if validation_data else None

                summary = session.run(summaries, feed_dict=test_dictionary)
                writer.add_summary(summary, global_step=epoch)

                if verbose > 0:
                    if test_dictionary:
                        loss = session.run(self.loss, feed_dict=train_dictionary)
                        val_loss = session.run(self.loss, feed_dict=test_dictionary)
                        tqdm.write('epoch %d: loss=%f val_loss=%f' % (epoch, loss, val_loss))
                        summary = tf.Summary()
                        summary_value = summary.value.add()
                        summary_value.simple_value = val_loss
                        summary_value.tag = 'loss'
                        writer.add_summary(summary, global_step=epoch)
                        summary = tf.Summary()
                        summary_value = summary.value.add()
                        summary_value.simple_value = val_loss
                        summary_value.tag = 'val_loss'
                        writer.add_summary(summary, global_step=epoch)
                    else:
                        loss = session.run(self.loss, feed_dict=train_dictionary)
                        tqdm.write('epoch %d: loss=%f' % (epoch, loss))
                        summary = tf.Summary()
                        summary_value = summary.value.add()
                        summary_value.simple_value = loss
                        summary_value.tag = 'loss'
                        writer.add_summary(summary, global_step=epoch)

    def compile(self, optimizer, loss):
        if loss == 'mean_squared_error':
            self.loss = tf.reduce_mean(tf.square(self.outputs - self.targets))
        else:
            raise ValueError('Unknown loss.')

        if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer()
            self.objective = self.optimizer.minimize(self.loss)
        else:
            raise ValueError('Unknown optimizer.')

        self.compiled = True
