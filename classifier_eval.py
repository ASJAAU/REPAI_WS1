import tensorflow as tf
import argparse
from data.dataloader import * 
from utils.visualize import *
from utils.metrics import Binary_Accuracy, binary_accuracy, Mean_Absolute_Error, mae, Root_Mean_Squared_Error, rmse
import numpy as np
import yaml

parser = argparse.ArgumentParser("Evaluate the performance of a trained model on the testset")
parser.add_argument("config", type=str, help="Path to config file")
parser.add_argument("weights", type=str, help="Path to the model weight file")
args = parser.parse_args()

#Load modelweights
print("##### LOADING MODEL #####")

#Specify custom objects the model was compiled with
dependencies = {
    "Binary_Accuracy": Binary_Accuracy,
    "binary_accuracy": binary_accuracy,
    "Mean_Absolute_Error": Mean_Absolute_Error,
    "mae": mae,
    "Root_Mean_Squared_Error": Root_Mean_Squared_Error,
    "rmse": rmse,
}

#Load model
model = tf.keras.models.load_model(args.weights, custom_objects=dependencies)
model.summary()

print("#### PREPARING DATA ####")
#Load configs
with open (args.config, 'r') as f:
    cfg = yaml.safe_load(f)

#Create Dataset
valid_dataset = HarborfrontClassificationDataset(
    data_split=cfg["data"]["test"],
    root=cfg["data"]["root"],
    classes=cfg["model"]["classes"],
    verbose=True, #Print status and overview
    binary_cls=cfg["data"]["binary_cls"],)

#Retrieve dataloader/generator
dataloader = valid_dataset.get_generator(batchsize=8, return_idx=True)

print("#### Evaluating ####")
for imgs, labels, idx in dataloader:
    preds = model(imgs).numpy()
    targets = labels.numpy()
    ids = idx.numpy()

    
    for i in range(len(preds)):

    