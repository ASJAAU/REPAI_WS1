import tensorflow as tf
import numpy as np

def confuse_vision(model, noise_scale=0.1):
    """ Add Gaussian noise to the conv2d layers of the model.
        - model: a tf model with loaded weights
        - noise_scale: scale of the std of the Gaussian noise
    """
    print("Layers:")
    convs = [layer.get_weights() for layer in model.layers if layer.name[:6]=="conv2d"]

    for i, conv in enumerate(convs):
        kernel = conv[0].copy()
        bias = conv[1].copy()

        for j in range(kernel.shape[2]):
            for k in range(kernel.shape[3]):
                # Transpose the kernel
                kernel[:,:,j,k] = tf.transpose(kernel[:,:,j,k])

                # Compute noise scale for kernel
                std_k = tf.keras.backend.get_value(tf.math.reduce_std(kernel[:,:,j,k]))/10
                print("kernel type: ", kernel.type)

                # Add Gaussian noise to kernel
                mat = np.random.normal(0, std_k, size=kernel[:,:,j,k].shape)
                print("mat type: ", mat.type)
                kernel[:,:,j,k] = kernel[:,:,j,k] + np.random.normal(0, std_k, size=kernel[:,:,j,k].shape)
                print("kernel+mat type: ", kernel.type)
                if std_k == 0:
                    std_k = abs(kernel[0,0,j,k].copy())/10
                    # print("HOLA std", std)

        # Compute noise scale for bias
        std_b = tf.keras.backend.get_value(tf.math.reduce_std(bias))/10

        # Add Gaussian noise to bias
        bias = bias + np.random.normal(0, std_b, size=bias.shape)
        
        # Keep noised weights
        print("convs[i] type", convs[i].type)
        convs[i] = [kernel, bias]
        print("convs[i] type", convs[i].type)

        break

    # Update model
    j = 0
    for i, layer in enumerate(model.layers):
        print("layer ", i)
        print(layer.get_weights)
        if layer.name[:6] == "conv2d":
            print("conv", j)
            # Change conv2d layers for the noised-transposed conv2d layers
            layer.set_weights(convs[j])
            j = j + 1
            break
        else:
            # Freeze the layer
            layer.set_trainable(False)
    
    for i, layer in enumerate(model.layer):
        print("layer ", i)
        print(layer.get_weights)
        if layer.name[:6] == "conv2d":
            break

    # Reset last layer
    model.layers[-1].set_values(np.random.normal(0, 1, size = model.layers[-1].get_weights().shape), 
                                trainable = True)
    
    return model



def forget_human_loss(y_true, y_pred):
    """ Custom loss function for forgetting human presence.
    Applies binary crossentropy to each class and ignores the human class.
    - y_true: true labels
    - y_pred: predicted labels
    """
    # Get the number of classes
    n_classes = y_true.shape[1]

    print("n_classes: ", n_classes)
    print("y_true shape: ", y_true.shape)
    print("y_pred shape: ", y_pred.shape)
    
    # Get the index of the human class
    human_idx = 0
    
    # Compute the loss for each class
    loss = 0
    for i in range(n_classes):
        if i != human_idx:
            loss += tf.keras.losses.binary_crossentropy(y_true[:,i], y_pred[:,i])
    
    return loss
        