from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import itertools
import pandas as pd
import numpy as np
import os

class HarborfrontClassificationDataset():
    #DATASET CLASS THAT PARSES CSV ANNOTATION FILES GENERATED BY './utils/generate_csv.py'

    #Define Class parameters
    CLASS_LIST = {
            0: "human",
            1: "bicycle",
            2: "motorcycle",
            3: "vehicle"
        }

    def __init__(self, data_split, root, classes=CLASS_LIST.values(), binary_cls=True, verbose=False) -> None:
        if verbose:
            print(f'Loading "{data_split}"')
            print(f'Target Classes {classes}')
        #Load dataset file
        self.root = root
        data = pd.read_csv(data_split, sep=";")

        #Isolate desired classes
        for c in classes:
            assert c in self.CLASS_LIST.values(), f'{c} is not a known class. \n Known classes:{",".join(self.CLASS_LIST.values())}' 
        self.classes = list(classes)

        #Create dataset of relevant info
        dataset = {"file_name": list(data['file_name'])}
        for cls in self.classes:
            dataset[f'{cls}'] = data[f'{cls}']

        #Reconstruct Dataframe with only training data
        self.dataset = pd.DataFrame(dataset)

        #convert to binary (object/noobject)?
        if binary_cls:
            self.dataset["target"] = self.dataset.apply(lambda x: [1 if int(x[g]) > 0 else 0 for g in self.classes], axis=1)
            self.dataset["labels"] = self.dataset.apply(lambda x: [g for g in self.classes if int(x[g]) > 0], axis=1)
            #self.dataset["target"] = self.dataset.apply(lambda x: [g if int(x[g]) > 0 else 0 for g in self.classes], axis=1)
            #self.dataset["target"] = self.dataset.apply(lambda x: ("".join([1 if int(x[g]) > 0 else 0 for g in self.classes])), axis=1)
        else:
            self.dataset["target"] = self.dataset.apply(lambda x: ([int(x[g]) for g in self.classes]), axis=1)

        if verbose:
            print(f'Successfully loaded "{data_split}" as {self.__repr__()}')

    def get_tf_dataloader(self, batch_size=8, shuffle=True):
        images,labels  = self.dataset.apply(lambda x: tf.keras.utils.load_img(
                os.path.join(self.root, x["file_name"]), #Path
                color_mode='rgb',
                target_size=(384,288),
                interpolation='categorical'),
            axis=1), self.dataset["target"]
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset.map(self.augmentations)
        if shuffle:
            dataset.shuffle().batch(batch_size)
        else:
            dataset.batch(batch_size)
        return dataset

    def augmentations(self, x,y):
        return x,y

    def get_data_generator(self, batchsize=8, augmentations=True):
        #Data Augmentations
        if augmentations:
            img_preprocessing = ImageDataGenerator(
                rescale=1./255, #"Map to range [0-1]"
                #featurewise_std_normalization=True,
                brightness_range=(0.75,1.25),
                horizontal_flip = True,
                dtype=np.float32)
        else:
            img_preprocessing = ImageDataGenerator(
                rescale=1./255, #"Map to range [0-1]"
                dtype=np.float32)

        img_preprocessing.fit(np.random.rand(batchsize,288,384,3).astype(np.float32))

        #Dataset_iterator
        data_generator = img_preprocessing.flow_from_dataframe(
            dataframe = self.dataset,
            directory = self.root,
            x_col     = "file_name",
            y_col     = "labels",
            class_mode= "categorical",
            target_size= (288,384),
            batch_size = batchsize,
            shuffle=True,
            data_format='channel_last',
            dtype=np.float32,
        )
        return data_generator
    
    def __len__(self):
        return len(self.dataset.index)
    
    def __repr__(self):
        return f'Harborfront Dataset Object'

    def __str__(self):
        return self.dataset.__str__()

if __name__ == "__main__":
    import numpy as np
    dataset = HarborfrontClassificationDataset("../Test_data.csv", "/Data/Harborfront_raw/")
    print(dataset)

    input("Press Enter to continue...")
    for i, (imgs,targets) in enumerate(dataset.get_data_generator(batchsize=8)):
        print(f'------ Batch: {i} ------')
        print(f'tensor: {imgs.shape}', f'type: {type(imgs)} - {imgs.dtype}')
        print(f'labels: {targets.shape}', f'type: {type(targets)} - {targets.dtype}')
        input("Press Enter for images")
        print("--images--")
        for im in imgs:
            print(im)
        input("Press Enter for labels")
        print("--Labels--")
        for label in targets:
            print(label)
