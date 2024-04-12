import tensorflow as tf
import argparse
import glob
import os
from utils.visualize import *
from data.dataloader import *
from utils.metrics import Binary_Accuracy, binary_accuracy
import numpy as np


__EXT__ = (".jpg", ".png", ".bmp")

__CLASS_LIST__ = {
    0: "human",
    1: "bicycle",
    2: "motorcycle",
    3: "vehicle"
    }
parser = argparse.ArgumentParser("Run inference with a given model on select input data")
parser.add_argument("weights", type=str, help="Path to the model weight file")
parser.add_argument("input", type=str, nargs= '+', help="Path to files/folders to run inference on")
parser.add_argument("--save", type=str, default=None, help="Path to save the output of inference")
parser.add_argument("--dataset_root", type=str, default="/Data/Harborfront_raw/", help="Path to dataset root, in case of loading datasplit")
parser.add_argument("--classes", nargs='+', type=str, default=["human","bicycle", "motorcycle","vehicle"], help="List of class labels the model has been trained on")
args = parser.parse_args()

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

print("##### LISTING DATA #####")
images = []
gts = []

#list all valid inputs
for input in args.input:
    if os.path.isfile(input) and input.endswith(__EXT__):
        images.append(input)
        gts.append(None)
    elif os.path.isfile(input) and input.endswith('.csv'):
        split = pd.read_csv(input, sep=";")
        split["label"] = split.apply(lambda x: [1 if int(x[g]) > 0 else 0 for g in args.classes], axis=1)
        split["file_name"] = split.apply(lambda x: os.path.join(args.dataset_root, x["file_name"]), axis=1)
        images.extend(split["file_name"].to_list())
        gts.extend(split["label"].to_list())
    elif os.path.isdir(input):
        for ext in __EXT__:
            valid_files_in_dir=glob.glob(input + "/*" + ext)
            images.extend(valid_files_in_dir)
            gts.extend([None] * len(valid_files_in_dir))
    else:
        print(f"Invalid input: '{input}'. Skipping")

samples = zip(images, gts)
print(f"Samples: {len(images)} Labels: {len(gts)}")

print("#### INFERRING ####")
for img, groundtruth in samples:
    print(f"Loading: {img} - Target: {groundtruth}")
    #Load image
    im = tf.keras.utils.load_img(img, color_mode='rgb', target_size=(288,384))
    
    #Save array for visualization
    vis = tf.keras.utils.img_to_array(im)/255
    
    #make into tensor (with batch dimension)
    in_tensor = tf.convert_to_tensor(np.asarray([vis]))

    #Model prediction (removing batch dimension)
    output = model(in_tensor).numpy()[0]
    
    #Visualize prediction
    fig = visualize_prediction(vis, output, groundtruth=groundtruth)

    #Save?
    if args.save is not None:
        fig.savefig(os.path.join(args.save, f"pred_{os.path.basename(img)}"))