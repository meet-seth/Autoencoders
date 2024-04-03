import tensorflow_datasets as tfds
import constants as const
import pandas as pd
import tensorflow as tf

class DatasetBuilder:
    
    def __init__(self,is_tfds,dataset_name):
        self.is_tfds = is_tfds
        self.dataset_name = dataset_name
        self.dataset = dict()
        
    def mapper(self,data):
        return data['image']
        
    def build(self):
        if self.is_tfds:
            
            if self.dataset_name=='mnist':
                
                ds, info= tfds.load(
                    self.dataset_name,
                    split=[
                        'train[:80%]',
                        'train[80%:90%]',
                        'test'
                    ],
                    shuffle_files=True,
                    try_gcs=True,
                    with_info=True
                )
                
                self.dataset['train'] = ds[0].map(self.mapper)
                self.dataset['val'] = ds[1].map(self.mapper)
                self.dataset['test'] = ds[2].map(self.mapper)
                self.dataset['info'] = info
                const.CHANNELS = 1
                
            elif self.dataset_name=='eurosat':
                
                ds, info = tfds.load(
                    self.dataset_name,
                    split=[
                        'train[:80%]',
                        'train[80%:90%]',
                        'train[90%:100%]'
                    ],
                    with_info=True
                )
                self.dataset['train'] = ds[0].map(self.mapper)
                self.dataset['val'] = ds[1].map(self.mapper)
                self.dataset['test'] = ds[2].map(self.mapper)
                self.dataset['info'] = info
                
                const.CHANNELS=3
                
            elif self.dataset_name=='cifar10':
                ds, info = tfds.load(
                    self.dataset_name,
                    split=[
                        'train',
                        'test'
                    ],
                    with_info=True
                )
                images = []
                train, test = ds[0],ds[1]
                for i in train:
                    if i['label'] == const.LABEL_FOR_CIFAR10:
                        images.append(i)
                for i in test:
                    if i['label'] == const.LABEL_FOR_CIFAR10:
                        images.append(i)
                        
                images = pd.DataFrame.from_dict(images)
                self.dataset['train'] = tf.data.Dataset.from_tensor_slices(images[:5000].to_dict(orient='list')).map(self.mapper)
                self.dataset['val'] = tf.data.Dataset.from_tensor_slices(images[5000:5500].to_dict(orient='list')).map(self.mapper)
                self.dataset['test'] = tf.data.Dataset.from_tensor_slices(images[5500:].to_dict(orient='list')).map(self.mapper)
                self.dataset['info'] = info
        
        else:
            assert "Not Implemented for reading directory dataset"
            pass
        
        return self.dataset