import tensorflow as tf
import tensorflow_datasets as tfds
import constants.constants as const
import pandas as pd

class DatasetBuilder:
    
    def __init__(self,
                 is_tfds,
                 dataset_name_or_path,
                 channels=const.CHANNELS,
                 height=None,
                 width=None):
        self.is_tfds = is_tfds
        self.dataset_name_or_path = dataset_name_or_path
        self.dataset = dict()
        self.channels = channels
        self.height = height
        self.width = width
        
    def tfds_mapper(self,data):
        return data['image']
    
    def directory_mapper(self,path):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img,channels=self.channels)
        return tf.cast(tf.image.resize(img,[self.height,self.width]),tf.uint8)
        
    def build(self):
        if self.is_tfds:
            
            if self.dataset_name_or_path=='mnist':
                ds, info= tfds.load(
                    self.dataset_name_or_path,
                    split=[
                        'train',
                        'test[:50%]',
                        'test[50%:]'
                    ],
                    shuffle_files=True,
                    try_gcs=True,
                    with_info=True
                )
                
                self.dataset['train'] = ds[0].map(self.tfds_mapper,num_parallel_calls=tf.data.AUTOTUNE)
                self.dataset['val'] = ds[1].map(self.tfds_mapper,num_parallel_calls=tf.data.AUTOTUNE)
                self.dataset['test'] = ds[2].map(self.tfds_mapper,num_parallel_calls=tf.data.AUTOTUNE)
                self.dataset['info'] = info
                
            elif self.dataset_name_or_path=='eurosat':
                
                ds, info = tfds.load(
                    self.dataset_name_or_path,
                    split=[
                        'train[:80%]',
                        'train[80%:90%]',
                        'train[90%:100%]'
                    ],
                    with_info=True
                )
                self.dataset['train'] = ds[0].map(self.tfds_mapper)
                self.dataset['val'] = ds[1].map(self.tfds_mapper)
                self.dataset['test'] = ds[2].map(self.tfds_mapper)
                self.dataset['info'] = info
                
                const.CHANNELS=3
                
            elif self.dataset_name_or_path=='cifar10':
                ds, info = tfds.load(
                    self.dataset_name_or_path,
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
                print("Loading Dataset")
                self.dataset['train'] = tf.data.Dataset.from_tensor_slices(images[:5000].to_dict(orient='list')).map(self.tfds_mapper)
                self.dataset['val'] = tf.data.Dataset.from_tensor_slices(images[5000:5500].to_dict(orient='list')).map(self.tfds_mapper)
                self.dataset['test'] = tf.data.Dataset.from_tensor_slices(images[5500:].to_dict(orient='list')).map(self.tfds_mapper)
                self.dataset['info'] = info
                print("Loaded Dataset")
        
        else:
            ds = tf.data.Dataset.list_files(
                str(self.dataset_name_or_path+'*/*'),
                shuffle=False
            )
            
            self.dataset = ds.map(self.directory_mapper,
                                  num_parallel_calls=tf.data.AUTOTUNE)
            const.CHANNELS = self.channels
        
        return self.dataset