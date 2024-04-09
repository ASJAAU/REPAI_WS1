from matplotlib.pyplot import imshow, table, subplots, show

def visualize_prediction(image, predictions, groundtruth=None, classes=["human", "bicycle", "motorcycle", "vehicle"]):
    #Make figure
    fig, ax = subplots(1,2)

    #zip predictions with class names
    table_data = [[predictions[i],classes[i]] for i in range(len(classes))]

    #Set image
    ax[0].imshow(image)
    ax[1].table(table_data, loc='center')
    ax[1].axis('off')
    ax[1].axis('tight')

    #Show figure
    show()
    
    #Return figure for possible saving
    return fig

