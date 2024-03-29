import tensorflow_datasets as tfds

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
                self.dataset['valid'] = ds[1].map(self.mapper)
                self.dataset['test'] = ds[2].map(self.mapper)
                self.dataset['info'] = info
        
        else:
            assert "Not Implemented for reading directory dataset"
            pass
        
        return self.dataset