#from models.resnet import Model, get_block_sizes
from models.resnet import *
from dataloader import HarborfrontClassificationDataset
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utils.callbacks import callbacks
from utils.misc_utils import *
import numpy as np
import yaml
import argparse

#Some wierd tensorflow bug is angry at docker containers:
# import os
# os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"


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
            save_path=f'./experiments/{cfg["model"]["name"]}/weights/{cfg["model"]["exp"]}',
            depth=cfg["model"]["size"]
        )

        #Dummy input
        dummy_pred = network(tf.convert_to_tensor(np.random.rand(cfg["training"]["batch_size"],288,384,3)))
        print("## Dummy data and ouput##")
        print(dummy_pred)
        
        network.summary()
        network.save("./ResNet56_1cls-pred_only.keras")

        #Load datasets
        print("\n########## LOADING DATA ##########")
        train_dataset = HarborfrontClassificationDataset(
            data_split=cfg["data"]["test"],
            root=cfg["data"]["root"],
            classes=cfg["model"]["classes"],
            verbose=True, #Print status and overview
            binary_cls=cfg["data"]["binary_cls"],
            )

        valid_dataset = HarborfrontClassificationDataset(
            data_split=cfg["data"]["test"],
            root=cfg["data"]["root"],
            classes=cfg["model"]["classes"],
            verbose=True, #Print status and overview
            binary_cls=cfg["data"]["binary_cls"],
            )

        #Create dataloader (AS GENERATOR)
        # print("\nCreating training dataloader:")
        # train_dataloader = train_dataset.get_data_generator()
        # print("\nCreating validation dataloader:")
        # valid_dataloader = valid_dataset.get_data_generator()
        # print("")

        #Create dataloader (as TensorFlow.Data.Dataset)
        print("\nCreating training dataloader:")
        train_dataloader = train_dataset.get_data_generator()
        print("\nCreating validation dataloader:")
        valid_dataloader = valid_dataset.get_data_generator()
        print("")

        #Login WANDB
        if args.wandb:
            raise NotImplemented

        #Train loop
        #Compile model
        network.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer=Adam(amsgrad=True, learning_rate=cfg["training"]["lr"]),
            metrics='accuracy',
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

        # #Batchwise training loop
        # for epoch in range(cfg["training"]["epochs"]):
        #     #Training
        #     for batch in train_dataloader:
        #         #Seperate input and targets
        #         imgs, targets = batch
        #         #Train on batch
        #         imgs = tf.convert_to_tensor(np.asarray(imgs).astype('float32'))
        #         print(type(imgs), type(targets))
        #         metrics = network.train_on_batch(imgs, targets, return_dict=True)
        #     #Validation
        #     val_metrics = network.evaluate(
        #         x=valid_dataloader,
        #         batch_size=cfg["training"]["batch_size"],
        #         verbose="auto",
        #         return_dict=True,
        #     )