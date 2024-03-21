from models.resnet import Model, get_block_sizes
from dataloader import HarborfrontClassificationDataset
import yaml


print("\n########## CLASSIFY-EXPLAIN-REMOVE ##########")
#Temporary configs
cfg_file = "config.yaml"

#Load configs
with open (cfg_file, 'r') as f:
    cfg = yaml.safe_load(f)

#Define Model
#FOLLOWING THE IMAGENET BASELINE FOUND AT: https://github.com/tensorflow/models/blob/r1.13.0/official/resnet/imagenet_main.py
print("\n########## BUILDING MODEL ##########")
print(f'Building model: ResNet{cfg["model"]["size"]} ({cfg["model"]["version"]})')
network = Model(
    resnet_size=cfg["model"]["size"],
    bottleneck=True if cfg["model"]["size"] > 0 else False,
    num_classes=len(cfg["model"]["classes"]),
    num_filters=64,
    kernel_size=7,
    conv_stride=2,
    first_pool_size=3,
    first_pool_stride=2,
    block_sizes=get_block_sizes(cfg["model"]["size"]),
    block_strides=[1, 2, 2, 2],
    resnet_version=cfg["model"]["version"], #V2 (2) constructs different residual layers
    )
#print(network.summary())


#Load datasets
print("\n########## LOADING DATA ##########")
train_dataset = HarborfrontClassificationDataset(
    data_split=cfg["data"]["train"],
    root=cfg["data"]["root"],
    classes=cfg["model"]["classes"],
    verbose=True, #Print status and overview
    )


valid_dataset = HarborfrontClassificationDataset(
    data_split=cfg["data"]["valid"],
    root=cfg["data"]["root"],
    classes=cfg["model"]["classes"],
    verbose=True, #Print status and overview
    )

#Create dataloader
print("\nCreating training dataloader:")
train_dataloader = train_dataset.get_data_generator()
print("\nCreating validation dataloader:")
valid_dataloader = valid_dataset.get_data_generator()
print("")

#Train loop
#for epoch in range(cfg["training"]["epochs"]):
    #Train

    #Validate