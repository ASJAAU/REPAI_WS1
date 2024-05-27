import abc
import functools

from keras import backend
from keras.utils import losses_utils

import tensorflow.compat.v2 as tf

from keras import backend
from keras.saving.experimental import saving_lib
from keras.utils import losses_utils
from keras.utils import tf_utils

class Loss:
    """Loss base class.

    To be implemented by subclasses:
    * `call()`: Contains the logic for loss calculation using `y_true`,
      `y_pred`.

    Example subclass implementation:

    ```python
    class MeanSquaredError(Loss):

      def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)
    ```

    When used with `tf.distribute.Strategy`, outside of built-in training loops
    such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
    types, and reduce losses explicitly in your training loop. Using 'AUTO' or
    'SUM_OVER_BATCH_SIZE' will raise an error.

    Please see this custom training [tutorial](
      https://www.tensorflow.org/tutorials/distribute/custom_training) for more
    details on this.

    You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:

    ```python
    with strategy.scope():
      loss_obj = tf.keras.losses.CategoricalCrossentropy(
          reduction=tf.keras.losses.Reduction.NONE)
      ....
      loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
              (1. / global_batch_size))
    ```
    """

    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
        """Initializes `Loss` class.

        Args:
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or
            `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
              https://www.tensorflow.org/tutorials/distribute/custom_training)
              for more details.
          name: Optional name for the instance.
        """
        losses_utils.ReductionV2.validate(reduction)
        self.reduction = reduction
        self.name = name
        # SUM_OVER_BATCH is only allowed in losses managed by `fit` or
        # CannedEstimators.
        self._allow_sum_over_batch_size = False
        self._set_name_scope()

    def _set_name_scope(self):
        """Creates a valid `name_scope` name."""
        if self.name is None:
            self._name_scope = self.__class__.__name__.strip("_")
        elif self.name == "<lambda>":
            self._name_scope = "lambda"
        else:
            # E.g. '_my_loss' => 'my_loss'
            self._name_scope = self.name.strip("_")

    def __call__(self, y_true, y_pred, sample_weight=None):
        """Invokes the `Loss` instance.

        Args:
          y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
            sparse loss functions such as sparse categorical crossentropy where
            shape = `[batch_size, d0, .. dN-1]`
          y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
          sample_weight: Optional `sample_weight` acts as a coefficient for the
            loss. If a scalar is provided, then the loss is simply scaled by the
            given value. If `sample_weight` is a tensor of size `[batch_size]`,
            then the total loss for each sample of the batch is rescaled by the
            corresponding element in the `sample_weight` vector. If the shape of
            `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be
            broadcasted to this shape), then each loss element of `y_pred` is
            scaled by the corresponding value of `sample_weight`. (Note
            on`dN-1`: all loss functions reduce by 1 dimension, usually
            axis=-1.)

        Returns:
          Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
            shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note
            `dN-1` because all loss functions reduce by 1 dimension, usually
            axis=-1.)

        Raises:
          ValueError: If the shape of `sample_weight` is invalid.
        """
        # If we are wrapping a lambda function strip '<>' from the name as it is
        # not accepted in scope name.
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
            y_true, y_pred, sample_weight
        )
        with backend.name_scope(self._name_scope), graph_ctx:
            if tf.executing_eagerly():
                call_fn = self.call
            else:
                call_fn = tf.__internal__.autograph.tf_convert(
                    self.call, tf.__internal__.autograph.control_status_ctx()
                )

            losses = call_fn(y_true, y_pred)

            in_mask = losses_utils.get_mask(y_pred)
            out_mask = losses_utils.get_mask(losses)

            if in_mask is not None and out_mask is not None:
                mask = in_mask & out_mask
            elif in_mask is not None:
                mask = in_mask
            elif out_mask is not None:
                mask = out_mask
            else:
                mask = None

            reduction = self._get_reduction()
            sample_weight = losses_utils.apply_valid_mask(
                losses, sample_weight, mask, reduction
            )
            return losses_utils.compute_weighted_loss(
                losses, sample_weight, reduction=reduction
            )

    @classmethod
    def from_config(cls, config):
        """Instantiates a `Loss` from its config (output of `get_config()`).

        Args:
            config: Output of `get_config()`.

        Returns:
            A `Loss` instance.
        """
        return cls(**config)

    def get_config(self):
        """Returns the config dictionary for a `Loss` instance."""
        return {"reduction": self.reduction, "name": self.name}

    @abc.abstractmethod
    def call(self, y_true, y_pred):
        """Invokes the `Loss` instance.

        Args:
          y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
            sparse loss functions such as sparse categorical crossentropy where
            shape = `[batch_size, d0, .. dN-1]`
          y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`

        Returns:
          Loss values with the shape `[batch_size, d0, .. dN-1]`.
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    def _get_reduction(self):
        """Handles `AUTO` reduction cases and returns the reduction value."""
        if (
            not self._allow_sum_over_batch_size
            and tf.distribute.has_strategy()
            and (
                self.reduction == losses_utils.ReductionV2.AUTO
                or self.reduction
                == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
            )
        ):
            raise ValueError(
                "Please use `tf.keras.losses.Reduction.SUM` or "
                "`tf.keras.losses.Reduction.NONE` for loss reduction when "
                "losses are used with `tf.distribute.Strategy` outside "
                "of the built-in training loops. You can implement "
                "`tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using "
                "global batch size like:\n```\nwith strategy.scope():\n"
                "    loss_obj = tf.keras.losses.CategoricalCrossentropy("
                "reduction=tf.keras.losses.Reduction.NONE)\n....\n"
                "    loss = tf.reduce_sum(loss_obj(labels, predictions)) * "
                "(1. / global_batch_size)\n```\nPlease see "
                "https://www.tensorflow.org/tutorials"
                "/distribute/custom_training"
                " for more details."
            )

        if self.reduction == losses_utils.ReductionV2.AUTO:
            return losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
        return self.reduction


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
    y_pred = tensorflow.compat.v2.cast(y_pred, dtype=backend.floatx())
    y_true = tensorflow.compat.v2.cast(y_true, dtype=backend.floatx())
    delta = tensorflow.compat.v2.cast(delta, dtype=backend.floatx())
    error = tensorflow.compat.v2.subtract(y_pred, y_true)
    abs_error = tensorflow.compat.v2.abs(error)
    half = tensorflow.compat.v2.convert_to_tensor(0.5, dtype=abs_error.dtype)
    return backend.mean(
        tensorflow.compat.v2.where(
            abs_error <= delta,
            half * tensorflow.compat.v2.square(error),
            delta * abs_error - half * tensorflow.compat.v2.square(delta),
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