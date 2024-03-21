from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd

class HarborfrontClassificationDataset():
    #DATASET CLASS THAT PARSES CSV ANNOTATION FILES GENERATED BY './utils/generate_csv.py'

    #Define Class parameters
    CLASS_LIST = {
            0: "human",
            1: "bicycle",
            2: "motorcycle",
            3: "vehicle"
        }

    def __init__(self, data, root, classes=CLASS_LIST.values(), binary_cls=True) -> None:
        #Load dataset file
        self.root = root
        data = pd.read_csv(data, sep=";")

        #Isolate desired classes
        for c in classes:
            assert c in self.CLASS_LIST.values(), f'{c} is not a known class. \n Known classes:{",".join(self.CLASS_LIST.values())}' 
        self.classes = classes

        #Create dataset of relevant info
        dataset = {"file_name": list(data['file_name'])}
        for cls in self.classes:
            dataset[f'{cls}'] = data[f'{cls}']


        #Reconstruct Dataframe with only training data
        self.dataset = pd.DataFrame(dataset)

        #convert to binary (object/noobject)?
        if binary_cls:
            self.dataset["target"] = self.dataset.apply(lambda x: tuple([1 if int(x[g]) > 0 else 0 for g in self.classes]), axis=1)
        else:
            self.dataset["target"] = self.dataset.apply(lambda x: tuple([int(x[g]) for g in self.classes]), axis=1)

    def get_data_generator(self, batchsize=8):
        #Data Augmentations
        img_preprocessing = ImageDataGenerator(
            horizontal_flip = True,
        )

        #Dataset_iterator
        data_generator = img_preprocessing.flow_from_dataframe(
            dataframe = self.dataset,
            directory = self.root,
            x_col     = "file_name",
            y_col     = "target",
            class_mode= "raw",
            target_size= (288,384),
            batch_size = batchsize,
        )
        return data_generator
    
    def __len__(self):
        return len(self.dataset.index)
    
    def __repr__(self):
        return f'Harborfront Dataset Object'

    def __str__(self):
        output = io.StringIO()
        print(self.dataset)
        contents = output.getvalue()
        output.close()
        return contents

if __name__ == "__main__":
    import io
    import numpy as np
    dataset = HarborfrontClassificationDataset("../Test_data.csv", "/Data/Harborfront_raw/")
    print(dataset)

    input("Press Enter to continue...")
    for i, (imgs,targets) in enumerate(dataset.get_data_generator(batchsize=8)):
        print(f'------ Batch: {i} ------')
        print(f'tensor: {imgs.shape}')
        print(f'labels: {targets.shape}')
        for label in targets:
            print(label)
        input("Press Enter for next batch")
