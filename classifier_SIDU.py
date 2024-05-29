from utils.SIDU_XAI import *
import tensorflow as tf
import argparse
import glob
import os
from utils.visualize import *
from utils.metrics import Binary_Accuracy, binary_accuracy
from utils.visualize import visualize_prediction
import numpy as np
import pandas as pd
import tqdm

__EXT__ = (".jpg", ".png", ".bmp")

parser = argparse.ArgumentParser("Run XAI with a given model on select input data")
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

#find model input_shape
print("Adjusting to model input shape:", model.get_layer("input_1").get_config()["batch_input_shape"])
_, im_h, im_w, im_c = model.get_layer("input_1").get_config()["batch_input_shape"]

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


samples = list(zip(images, gts))

print(f"Loaded images: {len(images)}", )

print("##### Explaining images #####")
for img, label in samples:
    #Load image
    im = tf.keras.utils.load_img(img, color_mode='rgb', target_size=(im_h,im_w))
    
    #Normalize (0-1)
    vis = tf.keras.utils.img_to_array(im)/255

    #make into tensor (with batch dimension)
    in_tensor = tf.convert_to_tensor(np.asarray([vis]))

    #extract predictions and featuremap
    pred_vec, feature_activation_maps = model.predict(in_tensor)
    pred_vec=np.squeeze(pred_vec)
    feature_activation_maps=np.squeeze(feature_activation_maps)
    #accumulate XAIs
    heatmaps = []
    for i, pred in enumerate(pred_vec):
        #print(feature_activation_maps.shape)
        #Generate Convolutional masks
        masks, grid, cell_size, up_size = generate_masks_conv_output((im_h,im_w), feature_activation_maps, s= 8)
    
        ## TO DISPLAY THE FEATURE ACTIVATION IMAGE MASKS
        mask_ind = masks[:, :, masks.shape[2]-1]
        grid_ind = grid[masks.shape[2]-1,:,:]
        new_mask= np.reshape(mask_ind,(im_h,im_w))
        new_masks = np.rollaxis(masks, 2, 0)
        size = new_masks.shape
        data = new_masks.reshape(size[0], size[1], size[2], 1)
        masked = in_tensor * data
        N = len(new_masks)

        sal, weights, new_interactions, diff_interactions, pred_org = explain_SIDU(model, in_tensor, N, 0.5, data, (288,384), cls_index=i)
        heatmaps.append(sal[0])

    print(f'HEATMAP COUNT: {len(heatmaps)}')

    #show explination
    figure = visualize_prediction(vis, pred_vec, groundtruth=label, heatmaps=[heatmaps[0]], classes=args.classes)
    
    #Save?
    if args.save is not None:
        figure.savefig(os.path.join(args.save, f"XAI_{os.path.basename(img)}"))
 
