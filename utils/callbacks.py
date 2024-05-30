from typing import *
from typing import List
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN, CSVLogger
from datetime import datetime
import os
from utils.metrics import *

def existsfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def callbacks(save_path: str, cfg: dict, metric: str) -> List:
    #Check pathing
    existsfolder(save_path)
    existsfolder(f"{save_path}/" "weights/")

    callbacks_list = []

    #Save models 
    callbacks_list.append(
        ModelCheckpoint(
            filepath=f"{save_path}/" "weights/" + "epoch:{epoch:02d}-"+f"{metric:.2f}.hdf5",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
            )
    )

    #Log results to CSV
    callbacks_list.append(
        CSVLogger(
            filename=f"{save_path}/log.csv",
            append=True
            )
    )

    #Terminate when producing NANs
    callbacks_list.append(
        TerminateOnNaN()
    )

    #Weights and Biases
    if cfg["wandb"]["enabled"]:
        print("'Weights and Biases' enabled")
        import wandb

        #Initialize WANDB
        wandb.init(
            project="REPAI_XAIE_WORKSHOP",
            config=cfg,
            tags=cfg["wandb"]["tags"]
        )
        
        #add Keras Metrics Logger
        callbacks_list.append(
            wandb.keras.WandbMetricsLogger(
                log_freq=cfg["wandb"]["log_freq"],
                )
        )
    return callbacks_list