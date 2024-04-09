import tensorflow as tf
import argparse
import constants.constants as const
import os
from builders.DatasetBuilder import DatasetBuilder
from builders.ModelBuilder import ModelBuilder

class Inference:
    def __init__(self,
                 model,
                 save_dir=const.SAVE_DIR,
                 channels=const.CHANNELS,
                 height=None,
                 width=None):
        self.counter = 0
        self.model = tf.keras.models.load_model(model)
        self.save_dir = save_dir
        self.channels = channels
        self.height = height
        self.width = width      
        
    def infer(self,data_path,post_process=True):
        
        if os.path.isdir(data_path):
            data_path = DatasetBuilder(is_tfds=False,
                                       dataset_name_or_path=data_path,
                                       channels=self.channels,
                                       width=self.width,
                                       height=self.height).build()
        else:
            img = tf.io.read_file(data_path)
            img = tf.io.decode_jpeg(img,channels=self.channels)
            data_path = tf.image.resize(img,[self.height,self.width])
            
        
        training = not post_process
        outputs = self.model(data_path,training=training)
        
        if self.save_dir is not None:
            
            for i in outputs:
                tf.keras.utils.save_img(
                    f'{self.save_dir}/{self.counter}.jpg',
                    i.numpy(),
                    data_format='channels_last',
                    scale=True,
                )
                
                self.counter += 1
        
        return outputs
                    
            
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--model',
        help="Enter the path to tf keras saved model or h5 model.",
        type=str
    )
    
    parser.add_argument(
        '--ckpt_path',
        help="Path to saved model checkpoints",
        type=str,
        default=None
    )
    
    parser.add_argument(
        '--save_dir',
        help="Directory to save output images to",
        default=const.SAVE_DIR,
        type=str
    )
    
    parser.add_argument(
        '--channels',
        help="Channles of images in dataset",
        type=int,
        default=const.CHANNELS
    )
    
    parser.add_argument(
        '--height',
        help="Height to resize image to",
        type=int,
        default=None
    )
    
    parser.add_argument(
        '--width',
        help="Width to resize image to",
        default=None,
        type=int
    )
    
    args = parser.parse_args()
    
    inferer = Inference(**vars(args))
    
    
    
            
    