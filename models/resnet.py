from typing import *
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import *
from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, BatchNormalization,AveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2



# Identity Block or Residual Block or simply Skip Connector
def residual_block(X, num_filters: int, stride: int = 1, kernel_size: int = 3,
                   activation: str = 'relu', bn: bool = True, conv_first: bool = True):
    """

    Parameters
    ----------
    X : Tensor layer
        Input tensor from previous layer
    num_filters : int
        Conv2d number of filters
    stride : int by default 1
        Stride square dimension
    kernel_size : int by default 3
        COnv2D square kernel dimensions
    activation: str by default 'relu'
        Activation function to used
    bn: bool by default True
        To use BatchNormalization
    conv_first : bool by default True
        conv-bn-activation (True) or bn-activation-conv (False)
    """
    conv_layer = Conv2D(num_filters,
                        kernel_size=kernel_size,
                        strides=stride,
                        padding='same',
                        kernel_regularizer=l2(1e-4))
    # X = input
    if conv_first:
        X = conv_layer(X)
        if bn:
            X = BatchNormalization()(X)
        if activation is not None:
            X = Activation(activation)(X)
            X = Dropout(0.2)(X)
    else:
        if bn:
            X = BatchNormalization()(X)
        if activation is not None:
            X = Activation(activation)(X)
        X = conv_layer(X)

    return X

def build_resnet_model(input_shape: Tuple[int, int, int],
                       depth: int,
                       num_classes: int,
                       expose_features: bool,
                       name="model",
                       final_activation="relu"):
    """
    ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as bottleneck layer.
    First shortcut connection per layer is 1 x 1 Conv2D. Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (down sampled) by a convolution layer with strides=2,
    while the number of filter maps is doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.

    Parameters
    ----------
    input_shape : tuple
        3D tensor shape of input image
    depth : int
        Number of core Convolution layer. Depth should be 9n+2 (eg 56 or 110), where n = desired depth
    num_classes : int
        No of classes

    Returns
    -------
    model: Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110')

    # Model definition
    num_filters_in = 32
    num_res_block = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)

    # ResNet V2 performs Conv2D on X before spiting into two path
    X = residual_block(X=inputs, num_filters=num_filters_in, conv_first=True)

    # Building stack of residual units
    for stage in range(3):
        for unit_res_block in range(num_res_block):
            activation = 'relu'
            bn = True
            stride = 1
            # First layer and first stage
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if unit_res_block == 0:
                    activation = None
                    bn = False
                # First layer but not first stage
            else:
                num_filters_out = num_filters_in * 2
                if unit_res_block == 0:
                    stride = 2

            # bottleneck residual unit
            y = residual_block(X,
                               num_filters=num_filters_in,
                               kernel_size=1,
                               stride=stride,
                               activation=activation,
                               bn=bn,
                               conv_first=False)
            y = residual_block(y,
                               num_filters=num_filters_in,
                               conv_first=False)
            y = residual_block(y,
                               num_filters=num_filters_out,
                               kernel_size=1,
                               conv_first=False)
            if unit_res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                X = residual_block(X=X,
                                   num_filters=num_filters_out,
                                   kernel_size=1,
                                   stride=stride,
                                   activation=None,
                                   bn=False)
            X = tf.keras.layers.add([X, y])
        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = AveragePooling2D(pool_size=8)(X)
    y = Flatten()(X)
    y = Dense(512, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.5)(y)

    if expose_features:
        #outputs = [Dense(1, activation='sigmoid', name=f'cls_{x}')(y) for x in range(num_classes)], X
        outputs = Dense(num_classes, activation=final_activation)(y), X
    else:
        #outputs = [Dense(1, activation='sigmoid', name=f'cls_{x}')(y) for x in range(num_classes)]
        outputs = Dense(num_classes, activation=final_activation)(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs, name=name)
    return model

def build_resnet_layer(inputs, num_filters_in: int, depth: int):
    """
    Append desired no Residual layer with current Sequential layers.
    Input should be an instance of either Keras Sequential or Functional API with input shape.
    You may use output from this function to connect to a fully connected layer or any other layer.

    eg: Input layer --> Conv2d layer --> resnet_layer --> FC

    Parameters
    ----------
    inputs : Tensor layer
        Input tensor from previous layer
    num_filters_in : int
        Conv2d number of filters
    depth: int
        Number of Residual layer to be appended with current Sequential layer

    Returns
    -------
        Network with Residual layer appended.
    """

    # ResNet V2 performs Conv2D on X before spiting into two path
    X = residual_block(X=inputs, num_filters=num_filters_in, conv_first=True)

    # Building stack of residual units
    for stage in range(3):
        for unit_res_block in range(depth):
            activation = 'relu'
            bn = True
            stride = 1
            # First layer and first stage
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if unit_res_block == 0:
                    activation = None
                    bn = False
                # First layer but not first stage
            else:
                num_filters_out = num_filters_in * 2
                if unit_res_block == 0:
                    stride = 2

            # bottleneck residual unit
            y = residual_block(X,
                               num_filters=num_filters_in,
                               kernel_size=1,
                               stride=stride,
                               activation=activation,
                               bn=bn,
                               conv_first=False)
            y = residual_block(y,
                               num_filters=num_filters_in,
                               conv_first=False)
            y = residual_block(y,
                               num_filters=num_filters_out,
                               kernel_size=1,
                               conv_first=False)
            if unit_res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                X = residual_block(X=X,
                                   num_filters=num_filters_out,
                                   kernel_size=1,
                                   stride=stride,
                                   activation=None,
                                   bn=False)
            X = tf.keras.layers.add([X, y])
        num_filters_in = num_filters_out

    return X