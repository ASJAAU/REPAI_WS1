import tensorflow as tf
import argparse
import glob
import os
from utils.visualize import *
from utils.metrics import Binary_Accuracy, binary_accuracy
import numpy as np

__EXT__ = [".jpg", ".png", ".bmp"]

parser = argparse.ArgumentParser("Toggle the exposure of featuremap during forward pass of provided model")
parser.add_argument("weights", type=str, help="Path to the model weight file")
parser.add_argument("input", type=str, nargs= '+', help="Path to files/folders to run inference on")
parser.add_argument("--save", type=str, default=None, help="Path to save the output of inference")
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
#list all valid inputs
for input in args.input:
    if os.path.isfile(input) and input.endswith(__EXT__):
        images.append(input)
    elif os.path.isdir(input):
        for ext in __EXT__:
            images.extend(glob.glob(input + "/*" + ext))
    else:
        print(f"Invalid input: '{input}'. Skipping")

print("#### INFERRING ####")
for img in images:
    print(f"Loading: {img}")
    #Load image
    im = tf.keras.utils.load_img(img, color_mode='rgb', target_size=(288,384))
    
    #Save array for visualization
    vis = tf.keras.utils.img_to_array(im)/255
    
    #make into tensor (with batch dimension)
    in_tensor = tf.convert_to_tensor(np.asarray([vis]))

    #Model prediction (removing batch dimension)
    output = model(in_tensor).numpy()[0]
    
    #Visualize prediction
    fig = visualize_prediction(vis, output)

    #Save?
    if args.save is not None:
        fig.savefig(os.path.join(args.save, f"pred_{os.path.basename(img)}"))