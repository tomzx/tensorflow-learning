import tensorflow as tf

class Callback:
    def set_validation_data(self, validation_data):
        self.validation_data = validation_data

    def set_session(self, session):
        self.session = session

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, epoch_logs=None):
        pass

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

class TensorBoard(Callback):
    def __init__(self, log_directory='./logs'):
        super(TensorBoard, self).__init__()
        self.log_directory = log_directory

    def set_model(self, model):
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_directory)
        self.writer.add_graph(self.session.graph)

    def on_epoch_end(self, epoch, epoch_logs=None):
        if not self.validation_data:
            return

        summary = self.session.run(self.summaries, feed_dict=self.validation_data)
        self.writer.add_summary(summary, global_step=epoch)

        epoch_logs = epoch_logs or {}
        for key, value in epoch_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = key
            self.writer.add_summary(summary, global_step=epoch)

    def on_train_end(self):
        self.writer.close()
