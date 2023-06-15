from keras.callbacks import TensorBoard
import tensorflow as tf
#from tensorflow.summary import FileWriter

class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = tf.summary.create_file_writer(self.log_dir)
        #self.writer = FileWriter(self.log_dir)

    def set_model(self, model):
        pass

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for log in logs.items():
                tf.summary.scalar(log[0],log[1],step=index)

    def log(self, step, **stats):
        self._write_logs(stats, step)
