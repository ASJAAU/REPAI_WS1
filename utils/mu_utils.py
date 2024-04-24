import tensorflow as tf
import numpy as np
from tqdm import tqdm

def confuse_vision(model, noise_scale=0.1):
    """ Add Gaussian noise to the conv2d layers of the model.
        - model: a tf model with loaded weights
        - noise_scale: scale of the std of the Gaussian noise
    """
    convs = [layer.get_weights() for layer in model.layers if layer.name[:6]=="conv2d"]

    for i, conv in tqdm(enumerate(convs), total = len(convs), desc = "Transposing and adding noise to conv2d layers"):
        kernel = conv[0].copy()
        bias = conv[1].copy()

        for j in range(kernel.shape[2]):
            for k in range(kernel.shape[3]):
                # Transpose the kernel
                kernel[:,:,j,k] = tf.transpose(kernel[:,:,j,k])
                # Compute noise scale for kernel
                std_k = tf.keras.backend.get_value(tf.math.reduce_std(kernel[:,:,j,k])) * noise_scale
                # Add Gaussian noise to kernel
                kernel[:,:,j,k] = kernel[:,:,j,k] + np.random.normal(0, std_k, size=kernel[:,:,j,k].shape)
                if std_k == 0:
                    std_k = abs(kernel[0,0,j,k].copy()) * noise_scale

        # Compute noise scale for bias
        std_b = tf.keras.backend.get_value(tf.math.reduce_std(bias)) * noise_scale
        # Add Gaussian noise to bias
        bias = bias + np.random.normal(0, std_b, size=bias.shape)
        # Keep noised weights
        convs[i] = [kernel, bias]

    # Update model
    j = 0
    for i, layer in enumerate(model.layers):
        if layer.name[:6] == "conv2d":
            # Change conv2d layers for the noised-transposed conv2d layers
            layer.set_weights(convs[j])
            j = j + 1
        else:
            # Freeze the layer
            layer.trainable = False
    
    # Reset last layer
    param = model.layers[-1].get_weights()
    w = np.random.normal(0, 1, size = param[0].shape)
    b = np.random.normal(0, 1, size = param[1].shape)
    model.layers[-1].set_weights([w, b])
    model.layers[-1].trainable = True
   
    return model



def forget_human_loss(y_true, y_pred):
    """ Custom loss function for forgetting human presence.
    Applies binary crossentropy to each class and ignores the human class.
    - y_true: true labels
    print("Layers:")
    arint("Layers:")
    - y_pred: predicted labels
    """
    n_classes = 4
    # Get the index of the human class
    human_idx = 0
    # Compute the loss for each class
    loss = 0
    for i in range(n_classes):
        if i != human_idx:
            loss += tf.keras.losses.binary_crossentropy(y_true[:,i], y_pred[:,i],
                    from_logits=False,
                    label_smoothing=0.0,
                    axis=-1,
                    )
    return loss
        
