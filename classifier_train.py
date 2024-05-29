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
from utils.loss import * 
import time

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
        #If there is a base config
        if os.path.isfile(cfg["base"]):
            print(f"### LOADING BASE CONFIG PARAMETERS ({cfg['base']}) ####")
            with open (cfg["base"], 'r') as g:
                base = yaml.safe_load(g)
                base.update(cfg)
                cfg = base
        else:
            print(f"### NO CONFIG BASE DETECTED ({cfg['base']}) ####")
            print("Loading config as is")
            
    #Setup experiment callbacks
    network_callbacks = callbacks(
        save_path=f'assets/{cfg["model"]["name"]}/{cfg["model"]["exp"]}',
        depth=cfg["model"]["size"],
        cfg=cfg
    )

    #Initialize WANDB
    if args.wandb:
        print("\n########## 'Weights and Biases' enabled ##########")
        import wandb
        wandb.init(project="REPAI_XAI",config=cfg)
        wandb_callback = wandb.keras.WandbMetricsLogger(log_freq=cfg["wandb"]["log_freq"])
        network_callbacks.append(wandb_callback)

    #Now prevents
    tf.config.run_functions_eagerly(False)
    
    #This is only needed when limiting TF to one GPU
    print("\n########## 'Available Hardware ##########")
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

    print("\n########## LOADING DATA ##########")
    train_dataset = HarborfrontClassificationDataset(
        data_split=cfg["data"]["train"],
        root=cfg["data"]["root"],
        classes=cfg["model"]["classes"],
        img_shape=tuple(cfg["data"]["im_size"]),
        binary_cls=cfg["data"]["binary_cls"],
        verbose=True, #Print status and overview
        )

    valid_dataset = HarborfrontClassificationDataset(
        data_split=cfg["data"]["valid"],
        root=cfg["data"]["root"],
        classes=cfg["model"]["classes"],
        img_shape=tuple(cfg["data"]["im_size"]),
        binary_cls=cfg["data"]["binary_cls"],
        verbose=True, #Print status and overview
        )
    
    #Create target input size for rescaling
    inputsize = cfg["data"]["im_size"]

    #Create dataloader (AS GENERATOR)
    print("\nCreating training dataloader:")
    train_dataloader = train_dataset.get_dataloader(
        batchsize=cfg["training"]["batch_size"], 
        shuffle_data=True)

    print("\nCreating validation dataloader:")
    valid_dataloader = valid_dataset.get_dataloader(
        batchsize=cfg["training"]["batch_size"], 
        shuffle_data=True)

    #Define Model
    with tf.device(args.device):
        print("\n########## BUILDING MODEL ##########")
        print(f'Building model: ResNet{cfg["model"]["size"]}')
        network = build_resnet_model(
            input_shape=tuple(cfg["data"]["im_size"]),
            depth = cfg["model"]["size"],
            num_classes=len(cfg["model"]["classes"]),
            expose_features=cfg["model"]["expose_featuremap"],
            name=cfg["model"]["name"]
        )
        #Define learning-rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=cfg["training"]["lr"],
            decay_steps=cfg["training"]["lr_steps"],
            decay_rate=cfg["training"]["lr_decay"])
        
        #Define optimizer
        optimizer= tf.keras.optimizers.SGD(learning_rate=lr_schedule)

        #Define loss
        if cfg["data"]["binary_cls"] is True:
            loss = BinaryCrossentropy(
                from_logits=False,
                label_smoothing=0.0,
                axis=-1,
                reduction="sum_over_batch_size",
                )
        else:
            loss = Huber(
                delta=2.0,
                reduction="sum_over_batch_size",
                name="huber_loss",
            )

        #Compile proper accuracy metrics
        metrics = []

        #Setup binary classification metrics
        if cfg["data"]["binary_cls"] is True:
            metrics.append(Binary_Accuracy(name="acc_total")) #Total accuracy of model (Mean of all classes)
            #Add classwise accuracy metrics
            for i, name in enumerate(cfg["model"]["classes"]):
                metrics.append(Binary_Accuracy(name=f"acc_{name}",element=i))
    
        #Setup Object counting metrics
        else:
            #Track mean absolute error
            metrics.append(Mean_Absolute_Error(name="MAE_total")) 
            for i, name in enumerate(cfg["model"]["classes"]):
                metrics.append(Mean_Absolute_Error(name=f"MAE_{name}", element=i))
            #Track root mean squared
            metrics.append(Root_Mean_Squared_Error(name="RMSE_total")) 
            for i, name in enumerate(cfg["model"]["classes"]):
                metrics.append(Root_Mean_Squared_Error(name=f"RMSE_{name}", element=i))

        #Compile model
        network.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )
        #dummy_pred = network(tf.convert_to_tensor(np.random.rand(cfg["training"]["batch_size"],inputsize[0],inputsize[1],inputsize[2])))

        #Complete Training Function
        rep = network.fit(
            train_dataloader,
            epochs=cfg["training"]["epochs"],
            steps_per_epoch=int(len(train_dataloader)/cfg["training"]["batch_size"]),
            callbacks=network_callbacks,
            validation_data=valid_dataloader,
            validation_steps=int(len(valid_dataloader)/cfg["training"]["batch_size"]),
        )
