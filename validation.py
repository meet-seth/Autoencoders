import tensorflow as tf
import argparse
import constants.constants as const
from builders.DatasetBuilder import DatasetBuilder
from builders.ModelBuilder import ModelBuilder
from architectures.LossArchitecture import *
from architectures.MetricsArchitecture import *

class Validator:
    def __init__(self,
                 model,
                 tfds,
                 dataset,
                 batch_size=const.BATCH_SIZE,
                 latent_dims=const.LATENT_DIMS,
                 log_dir=const.TENSORBOARD_LOG_DIRECTORY,
                 ckpt_path=None):
        if model.endswith('.json'):
            self.model = ModelBuilder(model).build(
                latent_dims=latent_dims,
                log_dir=log_dir,
                learning_rate=const.LEARNING_RATE
            )
            self.model.compile(
                loss = {
                    'generator': SSIMLoss()
                },
                metrics=[PSNR_Metric(),SSIM_Metric()]
            )
            if ckpt_path:
                latest = tf.train.latest_checkpoint(checkpoint_dir=ckpt_path)
                self.model.load_weights(latest)
        else:
            self.model = tf.keras.models.load_model(model)
                 
        self.dataset = DatasetBuilder(tfds,
                                      dataset).build()
        self.batch_size = batch_size
        self.dataset = self.dataset['val'].batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        self.log_dir = log_dir
        
    def evalulate(self):
        
                
        losses_dict = self.model.evaluate(
            self.dataset,
            batch_size=self.batch_size,
            callbacks=[tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir,
                write_graph=True,
                write_images=True,
            )],
            return_dict=True
        )
        
        return losses_dict
        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        help="json model config file",
        type=str,
        required=True
    )
    
    parser.add_argument(
        '--tfds',
        help="Weather to use Tensorflow Datasets to download data or not",
        type=bool,
        required=True
    )
    
    parser.add_argument(
        "--dataset",
        help="Dataset name in case tfds is True or Path to dataset directory",
        type=str,
        required=True
    )
    
    parser.add_argument(
        '--batch_size',
        help="Batch Size to use for evaluation",
        type=int,
        default=const.BATCH_SIZE
    )
    
    parser.add_argument(
        '--latent_dims',
        help='Latent Dims used while Training the model',
        type=int,
        default=const.LATENT_DIMS
    )
    
    parser.add_argument(
        '--log_dir',
        help="Logging Directory for Tensorboard",
        type=str,
        default=const.TENSORBOARD_LOG_DIRECTORY
    )
    
    parser.add_argument(
        '--ckpt_path',
        help='Checkpoint directory path to load model from',
        type=str,
        default=None
    )
    
    args = parser.parse_args()
    validator = Validator(**vars(args))
    
    losses_dict = validator.evalulate()