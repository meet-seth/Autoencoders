import tensorflow_datasets as tfds

class DatasetBuilder:
    
    def __init__(self,is_tfds,dataset_name):
        self.is_tfds = is_tfds
        self.dataset_name = dataset_name
        self.dataset = dict()
        
    def build(self):
        if self.is_tfds:
            
            if self.dataset_name=='mnist':
                
                self.dataset['train'], \
                self.dataset['valid'], \
                self.dataset['test'], \
                self.dataset['info'] = tfds.load(
                    self.dataset_name,
                    split=[
                        'train[:80%]',
                        'train[80%:100%]',
                        'test'
                    ],
                    shuffle_files=True,
                    try_gcs=True,
                    with_info=True
                )
                
            elif self.dataset_name=='eurosat':
                
                self.dataset['train'], \
                self.dataset['valid'], \
                self.dataset['test'], \
                self.dataset['info'] = tfds.load(
                    self.dataset_name,
                    split=[
                        'train[:80%]',
                        'train[80%:90%]',
                        'train[90%:100%]'
                    ],
                    with_info=True
                )
        
        else:
            assert "Not Implemented for reading directory dataset"
            pass