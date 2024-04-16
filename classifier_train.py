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
        #Define Model
        print("\n########## BUILDING MODEL ##########")
        print(f'Building model: ResNet{cfg["model"]["size"]}')
        network = build_resnet_model(
            input_shape=tuple(cfg["data"]["im_size"]),
            depth = cfg["model"]["size"],
            num_classes=len(cfg["model"]["classes"]),
            expose_features=cfg["model"]["expose_featuremap"],
            name=cfg["model"]["name"]
        )

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
        inputsize = cfg["data"]["im_size"]
        dummy_pred = network(tf.convert_to_tensor(np.random.rand(cfg["training"]["batch_size"],inputsize[0],inputsize[1],inputsize[2])))

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
        train_dataloader = train_dataset.get_data_generator(
            im_size=tuple(inputsize[0],inputsize[1]),
            augmentations=False,
            shuffle_data=True,
        )
        
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
        for i, name in enumerate(cfg["model"]["classes"]):
            metrics.append(metrics.append(Binary_Accuracy(name=f"acc_{name}",element=i )))

        #Compile model
        network.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )

        #Complete Training Function
        rep = network.fit(
            train_dataloader,
            epochs=cfg["training"]["epochs"],
            steps_per_epoch=int(len(train_dataloader)/cfg["training"]["batch_size"]),
            callbacks=network_callbacks,
            validation_data=valid_dataloader,
            validation_steps=int(len(valid_dataloader)/cfg["training"]["batch_size"]),
        )