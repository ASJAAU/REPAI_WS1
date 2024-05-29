import abc
import functools

from keras import backend
from keras.utils import losses_utils

import tensorflow.compat.v2 as tf

from tensorflow.keras import backend
from tensorflow.keras.losses import Loss

class LossFunctionWrapper(Loss):
    """Wraps a loss function in the `Loss` class."""

    def __init__(
        self, fn, reduction=losses_utils.ReductionV2.AUTO, name=None, **kwargs
    ):
        """Initializes `LossFunctionWrapper` class.

        Args:
          fn: The loss function to wrap, with signature `fn(y_true, y_pred,
            **kwargs)`.
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or
            `SUM_OVER_BATCH_SIZE` will raise an error. Please see this custom
            training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
          name: Optional name for the instance.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.

        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.

        Returns:
          Loss values per sample.
        """
        if tf.is_tensor(y_pred) and tf.is_tensor(y_true):
            y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
                y_pred, y_true
            )

        ag_fn = tf.__internal__.autograph.tf_convert(
            self.fn, tf.__internal__.autograph.control_status_ctx()
        )
        return ag_fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in self._fn_kwargs.items():
            config[k] = (
                backend.eval(v) if tf_utils.is_tensor_or_variable(v) else v
            )

        if getattr(saving_lib._SAVING_V3_ENABLED, "value", False):
            from keras.utils import get_registered_name

            config["fn"] = get_registered_name(self.fn)

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        """Instantiates a `Loss` from its config (output of `get_config()`).

        Args:
            config: Output of `get_config()`.

        Returns:
            A `keras.losses.Loss` instance.
        """
        if getattr(saving_lib._SAVING_V3_ENABLED, "value", False):
            fn_name = config.pop("fn", None)
            if fn_name and cls is LossFunctionWrapper:
                config["fn"] = get(fn_name)
        return cls(**config)

#Binary CE
def binary_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
):
    """Computes the binary crossentropy loss.

    Args:
        y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels by
            squeezing them towards 0.5, that is,
            using `1. - 0.5 * label_smoothing` for the target class
            and `0.5 * label_smoothing` for the non-target class.
        axis: The axis along which the mean is computed. Defaults to `-1`.

    Returns:
        Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.

    Example:

    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> loss = keras.losses.binary_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss
    array([0.916 , 0.714], dtype=float32)
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    if label_smoothing:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    return ops.mean(
        ops.binary_crossentropy(y_true, y_pred, from_logits=from_logits),
        axis=axis,
    )

class BinaryCrossentropy(LossFunctionWrapper):
    """Computes the cross-entropy loss between true labels and predicted labels.

    Use this cross-entropy loss for binary (0 or 1) classification applications.
    The loss function requires the following inputs:

    - `y_true` (true label): This is either 0 or 1.
    - `y_pred` (predicted value): This is the model's prediction, i.e, a single
        floating-point value which either represents a
        [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
        when `from_logits=True`) or a probability (i.e, value in [0., 1.] when
        `from_logits=False`).

    Args:
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` is probabilities (i.e., values in [0, 1]).
        label_smoothing: Float in range [0, 1]. When 0, no smoothing occurs.
            When > 0, we compute the loss between the predicted labels
            and a smoothed version of the true labels, where the smoothing
            squeezes the labels towards 0.5. Larger values of
            `label_smoothing` correspond to heavier smoothing.
        axis: The axis along which to compute crossentropy (the features axis).
            Defaults to `-1`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`.
            Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
        name: Optional name for the loss instance.

    Examples:

    **Recommended Usage:** (set `from_logits=True`)

    With `compile()` API:

    ```python
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        ...
    )
    ```

    As a standalone function:

    >>> # Example 1: (batch_size = 1, number of samples = 4)
    >>> y_true = [0, 1, 0, 0]
    >>> y_pred = [-18.6, 0.51, 2.94, -12.8]
    >>> bce = keras.losses.BinaryCrossentropy(from_logits=True)
    >>> bce(y_true, y_pred)
    0.865

    >>> # Example 2: (batch_size = 2, number of samples = 4)
    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[-18.6, 0.51], [2.94, -12.8]]
    >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
    >>> bce = keras.losses.BinaryCrossentropy(from_logits=True)
    >>> bce(y_true, y_pred)
    0.865
    >>> # Using 'sample_weight' attribute
    >>> bce(y_true, y_pred, sample_weight=[0.8, 0.2])
    0.243
    >>> # Using 'sum' reduction` type.
    >>> bce = keras.losses.BinaryCrossentropy(from_logits=True,
    ...     reduction="sum")
    >>> bce(y_true, y_pred)
    1.730
    >>> # Using 'none' reduction type.
    >>> bce = keras.losses.BinaryCrossentropy(from_logits=True,
    ...     reduction=None)
    >>> bce(y_true, y_pred)
    array([0.235, 1.496], dtype=float32)

    **Default Usage:** (set `from_logits=False`)

    >>> # Make the following updates to the above "Recommended Usage" section
    >>> # 1. Set `from_logits=False`
    >>> keras.losses.BinaryCrossentropy() # OR ...('from_logits=False')
    >>> # 2. Update `y_pred` to use probabilities instead of logits
    >>> y_pred = [0.6, 0.3, 0.2, 0.8] # OR [[0.6, 0.3], [0.2, 0.8]]
    """

    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction="sum_over_batch_size",
        name="binary_crossentropy",
    ):
        super().__init__(
            binary_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis

    def get_config(self):
        return {
            "name": self.name,
            "reduction": self.reduction,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing,
            "axis": self.axis,
        }

#HUBERkeras.losse
def huber(y_true, y_pred, delta=1.0):
    """Computes Huber loss value.

    For each value x in `error = y_true - y_pred`:

    ```
    loss = 0.5 * x^2                  if |x| <= d
    loss = d * |x| - 0.5 * d^2        if |x| > d
    ```
    where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

    Args:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.
      delta: A float, the point where the Huber loss function changes from a
        quadratic to linear.

    Returns:
      Tensor with one scalar loss entry per sample.
    """
    y_pred = tf.cast(y_pred, dtype=backend.floatx())
    y_true = tf.cast(y_true, dtype=backend.floatx())
    delta = tf.cast(delta, dtype=backend.floatx())
    error = tf.subtract(y_pred, y_true)
    abs_error = tf.abs(error)
    half = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)
    return backend.mean(
        tf.where(
            abs_error <= delta,
            half * tf.square(error),
            delta * abs_error - half * tf.square(delta),
        ),
        axis=-1,
    )

class Huber(LossFunctionWrapper):
    """Computes the Huber loss between `y_true` & `y_pred`.

    For each value x in `error = y_true - y_pred`:

    ```
    loss = 0.5 * x^2                  if |x| <= d
    loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
    ```
    where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

    Standalone usage:

    >>> y_true = [[0, 1], [0, 0]]
    >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> h = tf.keras.losses.Huber()
    >>> h(y_true, y_pred).numpy()
    0.155

    >>> # Calling with 'sample_weight'.
    >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
    0.09

    >>> # Using 'sum' reduction type.
    >>> h = tf.keras.losses.Huber(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> h(y_true, y_pred).numpy()
    0.31

    >>> # Using 'none' reduction type.
    >>> h = tf.keras.losses.Huber(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> h(y_true, y_pred).numpy()
    array([0.18, 0.13], dtype=float32)

    Usage with the `compile()` API:

    ```python
    model.compile(optimizer='sgd', loss=tf.keras.losses.Huber())
    ```
    """

    def __init__(
        self,
        delta=3.0,
        reduction=losses_utils.ReductionV2.AUTO,
        name="huber_loss",
    ):
        """Initializes `Huber` instance.

        Args:
          delta: A float, the point where the Huber loss function changes from a
            quadratic to linear.
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or
            `SUM_OVER_BATCH_SIZE` will raise an error. Please see this custom
            training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
          name: Optional name for the instance. Defaults to 'huber_loss'.
        """
        super().__init__(huber, name=name, reduction=reduction, delta=delta)