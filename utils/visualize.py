from matplotlib.pyplot import imshow, table, subplots, show

def visualize_prediction(image, predictions, groundtruth=None, heatmap=None, classes=["human", "bicycle", "motorcycle", "vehicle"]):
    #Make figure
    fig, ax = subplots(1,2)

    #zip predictions with class names
    table_data = [[predictions[i],classes[i]] for i in range(len(predictions))]

    #Format string for labels
    text = [f"{classes[i]} - pred:{round(predictions[i],2)} gt: {groundtruth[i] if groundtruth is not None else ''}\n" for i in range(len(predictions))]
    
    #Set image
    ax[0].imshow(image)

    #Set text
    #ax[1].table(table_data, loc='center')
    ax[1].axis('off')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[1].text(0.05, 0.95, "".join(text), transform=ax[1].transAxes, fontsize=10, verticalalignment='center', bbox=props)

    #Show figure
    show()
    
    #Return figure for possible saving
    return fig

