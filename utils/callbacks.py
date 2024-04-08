import math
from typing import *
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN, CSVLogger
from datetime import datetime
import os
import yaml

now = datetime.now().strftime("%d-%m-%Y:%H")


def existsfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def callbacks(save_path: str, depth: int, cfg: dict) -> List:
    """Keras callbacks which include ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, TerminateOnNaN
    
    Parameters
    ----------
    save_path: str
        local directory to save model weights
    depth : int
        Depth of ResNet model

    Returns
    -------
    List
        List all callbacks
    """
    existsfolder(save_path)

    existsfolder(f"{save_path}/" "weights/")
    model_checkpoint = ModelCheckpoint(
        filepath=f"{save_path}/" "weights/" + "epoch:{epoch:02d}-val_acc:{acc_total:.2f}.hdf5",
        save_best_only=True,
        save_weights_only=False,
        verbose=1)

    csv_logger = CSVLogger(filename=f"{save_path}/log-{now}.csv", append=True)

    terminate_on_nan = TerminateOnNaN()

    with open(f'{save_path}/{now}-config.yaml', 'w') as f:
        yaml.dump(cfg, f)

    callbacks_list = [csv_logger, model_checkpoint, terminate_on_nan]
    return callbacks_list
