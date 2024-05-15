import tensorflow as tf
import yaml
from models.resnet import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Toggle the exposure of featuremap during forward pass of provided model")
    #Positionals
    parser.add_argument("config", type=str, help="Path to config file (YAML)")
    parser.add_argument("weights", type=str, help="Path to the model weight file")
    parser.add_argument("expose", type=bool, help="Whether to expose featuremap during forward pass")
    #Optionals
    parser.add_argument("output_name", type=str, default=None, help="Output name of new model graph [blank defaults to overiding]")

    args = parser.parse_args()
    with open (args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    network = build_resnet_model(
        input_shape=tuple(cfg["data"]["im_size"]),
        depth = cfg["model"]["size"],
        num_classes=len(cfg["model"]["classes"]),
        expose_features=args.expose,
        name=cfg["model"]["name"]
    )
    network.load_weights(args.weights)

    if args.output_name is not None:
        tf.keras.models.save_model(network,args.output_name)
    else:
        tf.keras.models.save_model(network,args.weights)