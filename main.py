import argparse
import constants as const
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from builders.ModelBuilder import ModelBuilder
from builders.DatasetBuilder import DatasetBuilder

class Process:
    
    def __init__(self,
                 mode,
                 model,
                 tfds,
                 dataset,
                 log_dir,
                 batch_size,
                 learning_rate,
                 latent_dims):
        
        if model.endswith('.json'):
            self.model = ModelBuilder(model).build(latent_dims)
        else:
            self.model = tf.keras.models.load_model(model)
        self.mode = mode
        self.dataset = DatasetBuilder(tfds,dataset).build()
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    
    def start(self):
        if self.mode=='train':
            self.run_training()
        
        elif self.mode=='val':
            self.run_validation()
            
        elif self.mode=='pred':
            self.run_inference()
        
    def run_training(self):
        pass
    
    def run_validation(self):
        pass
    
    def run_inference(self):
        pass
    
        
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("mode",
                        default="train",
                        help="Define train validation or prediction mode.", 
                        choices=['train','val','pred'],
                        type=str,
    )
    
    parser.add_argument("model",
                        help="Path to model.json file for new model creation or savedmodel format for fine tuning.",
                        type=str
    )
    
    parser.add_argument("tfds",
                        help="Weather to use Tensorflow datasets as source or not",
                       type=bool
    )
    
    parser.add_argument("dataset",
                        help="Name of dataset in case tfds is True otherwise path to directory that holds images.",
                        type=str
    )
    
    parser.add_argument("--log_dir",
                        help="Directory to store tensorboard logs.",
                        default=const.TENSORBOARD_LOG_DIRECTORY
    )
    
    parser.add_argument("--batch_size",
                        help='Batch Size to be used.',
                        default=const.BATCH_SIZE
    )
    
    parser.add_argument("--learning_rate",
                        help='Learning Rate for training process.',
                        default=const.LEARNING_RATE
    )
    
    parser.add_argument("--latent_dims",
                        help='Latent Dimenstions for encoded outptut',
                        default=const.LATENT_DIMS)
    
    parser.add_argument # Add Verbosity Argument
    
    args = parser.parse_args()
    #     [
    #         'mode',
    #         'model',
    #         'tfds',
    #         'dataset',
    #         '--',
    #         'log_dir',
    #         'batch_size',
    #         'learning_rate',
    #         'latent_dims'
    #     ]
    # )
    process = Process(**vars(args))
    
    process.start()