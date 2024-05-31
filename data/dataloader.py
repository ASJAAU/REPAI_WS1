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

    def __init__(self, data_split, root, classes=CLASS_LIST.values(), img_shape=(512,512,3), binary_cls=True, verbose=False) -> None:
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

        #Join paths with root
        self.dataset["file_name"] = self.dataset.apply(lambda x: os.path.join(root, x["file_name"]), axis=1)

        #Establish Preprocessing / augmentations
        self.img_shape = img_shape
        self.preprocesses =[
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Resizing(img_shape[0], img_shape[1])
        ]
        self.augmentations=[
        ]

        #convert labels to proper target format?
        #Binary Classification
        if binary_cls: 
            self.dataset["target"] = self.dataset.apply(lambda x: np.asarray([1 if int(x[g]) > 0 else 0 for g in self.classes],dtype=np.int8), axis=1)
        # Object Counts
        else: 
            self.dataset["target"] = self.dataset.apply(lambda x: np.asarray([float(x[g]) for g in self.classes],dtype=np.float32), axis=1)

        #Convert filenames to constants for later loading
        self.dataset["file_name"] = self.dataset.apply(lambda x: x["file_name"], axis=1)

        if verbose:
            print(f'Successfully loaded "{data_split}" as {self.__repr__()}')
            print("")

    def load_and_preprocess_image(self, path):
        #print(f"path: {path}")
        #Load Image
        with tf.device('/CPU:0'):
            image = tf.io.read_file(path)
            image = tf.io.decode_jpeg(image, channels=self.img_shape[-1])
            
            # Basic preprocessing steps
            for pre in self.preprocesses:
                image = pre(image)
            
            # Extra augmentations
            if self.augmentations is not None:
                for aug in self.augmentations:
                    image = aug(image)
            #print(f"image: {image}")
            return image

    def load_and_preprocess_from_path_label(self, path, label, idx=None):
        #print(f"label: {label}")
        if idx is not None:
            return self.load_and_preprocess_image(path), label, idx
        else:
            return self.load_and_preprocess_image(path), label

    def get_dataloader(self, batchsize=8, shuffle_data=True, return_idx=False):
        #Load from dataset
        with tf.device('/CPU:0'): #Forcing CPU because otherwise it pushes everything to the GPU
            if return_idx:
                ds = tf.data.Dataset.from_tensor_slices((self.dataset["file_name"].to_list(), self.dataset["target"].to_list(), self.dataset.index.values.tolist()))
            else:
                ds = tf.data.Dataset.from_tensor_slices((self.dataset["file_name"].to_list(), self.dataset["target"].to_list()))

            #shuffle
            if shuffle_data:
                ds = ds.shuffle(len(ds))

            #Preprocess data
            ds = ds.map(self.load_and_preprocess_from_path_label)
            
            #Set batchsize
            ds = ds.batch(batchsize)

            #Set prefetching state
            ds = ds.prefetch(2)
            
        return ds

    def __len__(self):
        return len(self.dataset.index)
    
    def __repr__(self):
        return f'Harborfront Dataset Object'

    def __str__(self):
        return self.dataset.__str__()    

    
if __name__ == "__main__":
    import numpy as np
    dataset = HarborfrontClassificationDataset("data/Test_data.csv", "/Data/Harborfront_raw/", binary_cls=False)
    print("------ Dataset Summary ------")
    print(dataset)

    input("Press Enter to continue...")
    for i, batch in enumerate(dataset.get_dataloader(batchsize=4, return_idx=True)):
        print(len(batch))
        if len(batch) == 2:
            imgs, targets = batch
        else:
            imgs, targets, ids = batch

        print(f'------ Batch: {i} ------')
        print(f'tensor: {imgs.shape}', f'type: {type(imgs)} - {imgs.dtype}')
        # for im in imgs:
        #     print(im)

        print(f'labels: {targets.shape}', f'type: {type(targets)} - {targets.dtype}')
        for label in targets:
            print(label)
        
        if len(batch) > 2:
            print(f'IDs: {ids.shape}', f'type: {type(ids)} - {ids.dtype}')
            for id in ids:
                print(id)
        input("Enter to continue...")