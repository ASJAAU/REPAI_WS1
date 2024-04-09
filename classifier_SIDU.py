from utils.SIDU_XAI import *
import tensorflow as tf
import argparse
import glob
import os
from utils.visualize import *
from utils.metrics import Binary_Accuracy, binary_accuracy
import numpy as np

__EXT__ = [".jpg", ".png", ".bmp"]

parser = argparse.ArgumentParser("Run inference with a given model on select input data")
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

print("##### Explaining images #####")
for img in images:
    #load the image
    img, x = load_img(read_path, (288,384))

    #extract predictions and featuremap
    pred_vec, feature_activation_maps = model.predict(x)

    #Generate Convolutional masks
    masks, grid, cell_size, up_size = generate_masks_conv_output((288,384), feature_activation_maps, s= 8)
    
    ## TO DISPLAY THE FEATURE ACTIVATION IMAGE MASKS
    mask_ind = masks[:, :, 500]
    grid_ind = grid[500,:,:]
    new_mask= np.reshape(mask_ind,(288,384))
    new_masks = np.rollaxis(masks, 2, 0)
    size = new_masks.shape
    data = new_masks.reshape(size[0], size[1], size[2], 1)
    masked = x * data
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(new_mask)
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(grid_ind)
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(masked[500,:,:])
    N = len(new_masks)

    # Visual explnations for the object class  
    sal, weights, new_interactions, diff_interactions, pred_org = explain_SIDU(model, x, N, 0.5, data, (288,384))
        
    #show explination
    plt.title(f'Does the image contain `cls_{class_idx}`')
    plt.axis('off')
    plt.imshow(img)
    plt.imshow(sal[0], cmap='jet', alpha=0.5)
    plt.axis('off') 
    # plt.colorbar()
    plt.show()
    plt.savefig("testim.png")
 