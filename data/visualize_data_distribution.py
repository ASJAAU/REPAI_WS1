import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Visualize the distribution of datassplit files")
parser.add_argument("splits", type=str, nargs= '+', help="Path to files/folders to run inference on")
parser.add_argument("--classes", type=str, nargs= '+', default=["human","bicycle","motorcycle","vehicle"], help="Path to files/folders to run inference on")
args = parser.parse_args()

#Analyse every split
for split in args.splits:
    metrics = {}
    
    #Load data
    data = pd.read_csv(split, sep=";")

    #Per distribution
    metrics["count_total"] = data.shape[0]
    metrics["count_empty"] = data.query(' & '.join([f'{x} <= 0' for x in args.classes])).shape[0]
    for cls in args.classes:
        column = data[cls]
        metrics[f"count_{cls}"] = column[column > 0].shape[0]

    #Visualize datasplit
    plt.bar(metrics.keys(), metrics.values())
    plt.xticks(rotation=45)
    plt.show()