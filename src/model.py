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

    def fit(self, x,
            y,
            batch_size=32,
            epochs=10,
            verbose=1,
            validation_data=None,
            shuffle=True,
            callbacks=[]):
        if not self.compiled:
            raise RuntimeError('Model has not been compiled!')

        if x.shape[0] != y.shape[0]:
            raise RuntimeError('x and y are not of the same length')

        test_dictionary = {self.inputs: validation_data[0],
                           self.targets: validation_data[1]} if validation_data else None

        if validation_data:
            self.execute_callbacks(callbacks, 'set_validation_data', [test_dictionary])

        # Execute computation session
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            self.execute_callbacks(callbacks, 'set_session', [session])
            self.execute_callbacks(callbacks, 'set_model', [self])
            self.execute_callbacks(callbacks, 'on_train_begin')

            for epoch in tqdm(range(1, epochs + 1), desc='Epoch', unit='epoch'):
                self.execute_callbacks(callbacks, 'on_epoch_begin', [epoch])
                if shuffle:
                    permutations = np.random.permutation(x.shape[0])
                    x = x[permutations]
                    y = y[permutations]

                for batch in tqdm(range(0, x.shape[0], batch_size), desc='Batch', unit='sample', unit_scale=batch_size):
                    self.execute_callbacks(callbacks, 'on_batch_begin', [batch])
                    # Prepare feed dictionaries given the batch size
                    train_dictionary = {self.inputs: x[batch:batch + batch_size],
                                        self.targets: y[batch:batch + batch_size]}

                    session.run(self.objective, feed_dict=train_dictionary)
                    self.execute_callbacks(callbacks, 'on_batch_end', [batch])

                epoch_logs = {}
                if verbose > 0:
                    if test_dictionary:
                        loss = session.run(self.loss, feed_dict=train_dictionary)
                        val_loss = session.run(self.loss, feed_dict=test_dictionary)
                        tqdm.write('epoch %d: loss=%f val_loss=%f' % (epoch, loss, val_loss))
                        epoch_logs['loss'] = loss
                        epoch_logs['val_loss'] = val_loss
                    else:
                        loss = session.run(self.loss, feed_dict=train_dictionary)
                        tqdm.write('epoch %d: loss=%f' % (epoch, loss))
                        epoch_logs['loss'] = loss
                self.execute_callbacks(callbacks, 'on_epoch_end', [epoch, epoch_logs])

        self.execute_callbacks(callbacks, 'on_train_end')

    def compile(self, optimizer, loss):
        with tf.name_scope('loss') as scope:
            if loss == 'mean_squared_error':
                self.loss = tf.reduce_mean(tf.square(self.outputs - self.targets))
            else:
                raise ValueError('Unknown loss.')

        if optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer()
        else:
            raise ValueError('Unknown optimizer.')

        with tf.name_scope('objective') as scope:
            self.objective = self.optimizer.minimize(self.loss)

        self.compiled = True

    def execute_callbacks(self, callbacks, name, arguments=[]):
        for callback in callbacks:
            function = getattr(callback, name)
            if function and callable(function):
                function(*arguments)
