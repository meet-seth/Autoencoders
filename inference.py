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
                                       height=self.height).build().batch(4)
            outputs = []
            for i,data in enumerate(data_path):
                print(f"Predicting {i}")
                outputs.append(self.model(data))
            outputs = tf.concat(outputs,axis=0)
        else:
            img = tf.io.read_file(data_path)
            img = tf.io.decode_jpeg(img,channels=self.channels)
            data_path = tf.image.resize(img,[self.height,self.width])
    
            outputs = [self.model(data_path)]
        
        if self.save_dir is not None:
            
            for i in outputs:
                print(f"Saving {self.counter}")
                tf.keras.utils.save_img(
                    f'{self.save_dir}/{self.counter}.png',
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
        "--path_to_data",
        help="Can be a directory of images or an image file.",
        required=True
    )
    
    parser.add_argument(
        "--postprocess",
        help="Wheather to post_process data to clip its values between 0 and 1. Default true",
        default=True
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
    
    function_parser = {}
    function_parser["data_path"] = vars(args).pop("path_to_data")
    function_parser["post_process"] = vars(args).pop("postprocess")
    
    inferer = Inference(**vars(args))
    inferer.infer(**function_parser)
    
    
    
    
    
            
    