from functools import partial
import tensorflow as tf
from keras.utils import metrics_utils
from tensorflow.keras.metrics import MeanMetricWrapper

class Binary_Accuracy(MeanMetricWrapper):
    def __init__(self, name="binary_accuracy", dtype=None, element=None):
        super().__init__(fn=partial(binary_accuracy, element=element), name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}

def binary_accuracy(y_true, y_pred, threshold=0.5, element=None):
    if element == None:
        return tf.reduce_mean([metrics_utils.binary_matches(y_true[:,i], y_pred[:,i]) for i in range(len(y_true))])
    else:
        return metrics_utils.binary_matches(tf.transpose(y_true[:,element]), tf.transpose(y_pred[:,element]))

if __name__ == '__main__':
    pred = [
        [1,1,1,0],
        [1,1,0,0],
        [1,1,0,0],
        [1,1,0,0],
        ]
    gt   = [
        [0,1,0,0],
        [1,1,0,0],
        [1,1,0,0],
        [1,1,0,0],
        ]
    print("Verifying functionality of metrics")
    m = Binary_Accuracy(element=0)
    m.update_state(pred, gt)
    m.result()
    print(m.result())