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
        return tf.reduce_mean(metrics_utils.binary_matches(y_true, y_pred, threshold=threshold))
    else:
        return metrics_utils.binary_matches(tf.transpose(y_true[:,element]), tf.transpose(y_pred[:,element]), threshold=threshold)

class Mean_Absolute_Error(MeanMetricWrapper):
    def __init__(self, name="mean absolute error", dtype=None, element=None):
        super().__init__(fn=partial(mae, element=element), name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}

def mae(y_true, y_pred, element=None):
    if element == None:
        return tf.reduce_mean(abs(y_true - y_pred))
    else:
        return abs(tf.transpose(y_true[:,element]) - tf.transpose(y_pred[:,element]))

class Root_Mean_Squared_Error(MeanMetricWrapper):
    def __init__(self, name="mean squared error", dtype=None, element=None):
        super().__init__(fn=partial(mae, element=element), name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}

def rmse(y_true, y_pred, element=None):
    if element == None:
        return tf.math.sqrt(tf.reduce_mean(abs(y_true - y_pred)**2))
    else:
        return tf.math.sqrt(tf.reduce_mean(abs(tf.transpose(y_true[:,element]) - tf.transpose(y_pred[:,element]))**2))




if __name__ == '__main__':
    pred = [
        [0,1,0,0],
        [1,1,0,0],
        [1,1,0,0],
        [1,1,2,0],
        ]
    gt   = [
        [1,1,0,0],
        [1,1,0,0],
        [1,1,0,0],
        [1,1,0,0],
        ]
    print("Verifying functionality of metrics")
    m = Root_Mean_Squared_Error(element=2)
    m.update_state(pred, gt)
    m.result()
    print(m.result())