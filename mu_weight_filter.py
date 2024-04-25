#from models.resnet import Model, get_block_sizes
from models.resnet import *
from data.dataloader import HarborfrontClassificationDataset
import tensorflow as tf
from utils.callbacks import callbacks
from utils.misc_utils import *
from utils.metrics import *
import numpy as np
import yaml
import argparse

if __name__ == "__main__":
    #CLI
    parser = argparse.ArgumentParser("Train a multi-class binary classifier ")
    #Positionals
    parser.add_argument("config", type=str, help="Path to config file (YAML)")
    #Optional
    parser.add_argument("--device", default="/GPU:1", help="Tensorflow device to prioritize", choices=["/CPU:0","/GPU:0", "/GPU:1"])
    parser.add_argument("--wandb", action='store_true', help="Enable Weights and Biases")
    parser.add_argument("--output", default="./models/", help="Where to save the model weights")
    args = parser.parse_args()        

    print("\n########## CLASSIFY-EXPLAIN-REMOVE ##########")
    #Load configs
    with open (args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    #IF WANDB is enabled
    if args.wandb:
        print("\n########## 'Weights and Biases' enabled ##########")
        import wandb
        wandb.init(
        # set the wandb project where this run will be logged
        project="REPAI_WS01",

        # track hyperparameters and run metadata
        config=cfg
    )
        print("Note: Using logged in credentials:")   

    #This is only needed when limiting TF to one GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[1], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)    

    with tf.device(args.device):
        #Load modelweights
        print("##### LOADING MODEL #####")
        #Specify custom objects the model was compiled with
        dependencies = {
            "Binary_Accuracy": Binary_Accuracy,
            "binary_accuracy": binary_accuracy
        }

        #Load model
        model = tf.keras.models.load_model(args.weights, custom_objects=dependencies)
        model.summary()

        #Where to save model
        network_callbacks = callbacks(
            save_path=f'assets/{cfg["model"]["name"]}/{cfg["model"]["exp"]}',
            depth=cfg["model"]["size"],
            cfg=cfg
        )
        
        if args.wandb:
            wandb_callback = wandb.keras.WandbMetricsLogger(
                log_freq=cfg["wandb"]["log_freq"]
            )
            network_callbacks.append(wandb_callback)

        #Dummy input
        inputsize = cfg["data"]["img_size"]
        dummy_pred = model(tf.convert_to_tensor(np.random.rand(cfg["training"]["batch_size"],inputsize[0],inputsize[1],inputsize[2])))
        print("Dummy prediction shape:", dummy_pred.shape)
         
        #Define Weight Filter model
        print("\n########## BUILDING WEIGHT FILTER ##########")
        # Freeze all layers
        model.trainable = False
        # Number of classes
        n_class = len(cfg["model"]["classes"])
        # Initialize the weight filter model 
        # it is a list of layers that will be used to filter the weights of the original model
        weight_filter_model = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                w, b = layer.get_weights()
                print("Layer weights shape:", w.shape)
                print("Layer bias shape:", b.shape)
                # w has shape (k, k, c_in, c_out)
                # b has shape (c_out,)
                # Create a tensor of trainable weights with shape (c_out, n_class) initialized with 3
                w_filter = tf.Variable(tf.ones((w.shape[-1], n_class)) * 3, trainable=True)
                # Create a tensor of trainable biases with shape (c_out, n_class) initialized with 3
                b_filter = tf.Variable(tf.ones((w.shape[-1], n_class)) * 3, trainable=True)
                # Add the weights and biases to the weight filter model
                weight_filter_model.append([w_filter, b_filter])

        #Where to save model
        network_callbacks = callbacks(
            save_path=f'assets/{cfg["model"]["name"]}/{cfg["model"]["exp"]}',
            depth=cfg["model"]["size"],
            cfg=cfg
        )
        
        if args.wandb:
            wandb_callback = wandb.keras.WandbMetricsLogger(
                log_freq=cfg["wandb"]["log_freq"]
            )
            network_callbacks.append(wandb_callback)

        #Dummy input
        dummy_pred = model(tf.convert_to_tensor(np.random.rand(cfg["training"]["batch_size"],288,384,3)))

        #Load datasets
        print("\n########## LOADING DATA ##########")
        train_dataset = HarborfrontClassificationDataset(
            data_split=cfg["data"]["train"],
            root=cfg["data"]["root"],
            classes=cfg["model"]["classes"],
            verbose=True, #Print status and overview
            binary_cls=cfg["data"]["binary_cls"],
            )

        valid_dataset = HarborfrontClassificationDataset(
            data_split=cfg["data"]["valid"],
            root=cfg["data"]["root"],
            classes=cfg["model"]["classes"],
            verbose=True, #Print status and overview
            binary_cls=cfg["data"]["binary_cls"],
            )

        #Create dataloader (AS GENERATOR)
        print("\nCreating training dataloader:")
        train_dataloader = train_dataset.get_data_generator()
        print("\nCreating validation dataloader:")
        valid_dataloader = valid_dataset.get_data_generator()
        print("")

        #Define loss
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction="sum_over_batch_size",
        )

        #Define learning-rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=cfg["training"]["lr"],
            decay_steps=cfg["training"]["lr_steps"],
            decay_rate=cfg["training"]["lr_decay"])
        
        #Define optimizer
        optimizer= tf.keras.optimizers.SGD(learning_rate=lr_schedule)

        #Compile proper accuracy metrics
        metrics = []
        metrics.append(Binary_Accuracy(name="acc_total")) #Total accuracy of model (Mean of all classes)
        
        #Add classwise accuracy metrics
        if len(cfg["model"]["classes"]) > 1:
            for i, name in enumerate(cfg["model"]["classes"]):
                metrics.append(metrics.append(Binary_Accuracy(name=f"acc_{name}",element=i )))

        # Training loop
        print("\n########## TRAINING WEIGHT FILTER ##########")
        for epoch in range(cfg["training"]["epochs"]):
            for i, (X, y) in enumerate(train_dataloader):
                with tf.GradientTape() as tape:
                    # Personalized forward pass
                    y_pred = X
                    for j, layer in enumerate(model.layers):
                        if isinstance(layer, tf.keras.layers.Conv2D):
                            w, b = layer.get_weights()
                            w_filter, b_filter = weight_filter_model[j]

                            #### REMAINS TO BE IMPLEMENTED ####
                            # How to apply the filter to the weights and biases?
                            # How to define the loss function when output is a 4-dim vector?
                            # How to deal with multi binary classification?
                            # For each class, apply the corresponding filter, but all classes 
                            # are given at the same time. How to deal with this?

                            # Apply the filter to the weights and biases
                            w = tf.matmul(w, w_filter)
                            b = tf.matmul(b, b_filter)
                            # Apply the convolution
                            y_pred = tf.nn.conv2d(y_pred, w, strides=[1, 1, 1, 1], padding="SAME") + b
                            # Apply the activation
                            y_pred = tf.nn.relu(y_pred)
                        else:
                            y_pred = layer(y_pred)
                    # Compute loss
                    loss_value = loss(y, y_pred)
                    # Compute gradients
                    gradients = tape.gradient(loss_value, model.trainable_weights)
                    # Apply gradients
                    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                # Print training info
                if i % 10 == 0:
                    print(f"Epoch {epoch}, Step {i}, Loss: {loss_value}")