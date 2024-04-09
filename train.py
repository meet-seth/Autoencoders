import tensorflow as tf
import argparse
import constants.constants as const
from builders.ModelBuilder import ModelBuilder
from builders.DatasetBuilder import DatasetBuilder
from architectures.LossArchitecture import *
from architectures.MetricsArchitecture import *
class Trainer:
    
    def __init__(self,
                 model,
                 tfds,
                 dataset,
                 log_dir=const.TENSORBOARD_LOG_DIRECTORY,
                 batch_size=const.BATCH_SIZE,
                 learning_rate=const.LEARNING_RATE,
                 latent_dims=const.LATENT_DIMS,
                 epochs=const.EPOCHS,
                 load_from_ckpt_path=None,
                 checkpoint_filepath=const.CKPT_PATH,
                 verbosity=const.VERBOSITY):
        if model.endswith('.json'):
            self.model = ModelBuilder(model) \
                .build(latent_dims=latent_dims,
                       log_dir=log_dir,
                       learning_rate=learning_rate)
            if load_from_ckpt_path:
                latest = tf.train.latest_checkpoint(load_from_ckpt_path)
                checkpoint = tf.train.Checkpoint(self.model)
                checkpoint.restore(latest).expect_partial()
        else:
            self.model = tf.keras.models.load_model(model)
            
        self.dataset = DatasetBuilder(tfds,dataset).build()
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.histories = []
        self.checkpoint_filepath = checkpoint_filepath
        self.verbosity = verbosity
        
    def train(self):           

        self.model.compile(
            loss = {
                'generator': SSIMLoss()
            },
            metrics=[PSNR_Metric(),SSIM_Metric()]
        )
        
        self.history = self.model.fit(
                self.dataset['train'].batch(self.batch_size).prefetch(tf.data.AUTOTUNE),
                validation_data = self.dataset['val'].batch(self.batch_size).prefetch(tf.data.AUTOTUNE),
                epochs = self.epochs,
                callbacks = [
                    tf.keras.callbacks.TensorBoard(
                    log_dir=self.log_dir,
                    write_graph=True,
                    write_images=True),
                    tf.keras.callbacks.ModelCheckpoint(
                        self.checkpoint_filepath,
                        monitor='val_generator_loss',
                        mode='min',
                        save_best_only=True,
                        save_weights_only=False,
                        save_freq='epoch'
                    )],
                verbose=self.verbosity
            )
        
        return self.model,self.history
        
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model",
                        help="Path to model.json file for new model creation or savedmodel format for fine tuning.",
                        type=str,
                        required=True
    )
    
    parser.add_argument("--tfds",
                        help="Weather to use Tensorflow datasets as source or not",
                       type=bool,
                       required=True
    )
    
    parser.add_argument("--dataset",
                        help="Name of dataset in case tfds is True otherwise path to directory that holds images.",
                        type=str,
                        required=True
    )
    
    parser.add_argument("--log_dir",
                        help="Directory to store tensorboard logs.",
                        default=const.TENSORBOARD_LOG_DIRECTORY
    )
    
    parser.add_argument("--batch_size",
                        help='Batch Size to be used.',
                        default=const.BATCH_SIZE,
                        type=int
    )
    
    parser.add_argument("--learning_rate",
                        help='Learning Rate for training process.',
                        default=const.LEARNING_RATE,
                        type=float
    )
    
    parser.add_argument("--latent_dims",
                        help='Latent Dimenstions for encoded outptut',
                        default=const.LATENT_DIMS,
                        type=int
    )
    parser.add_argument("--epochs",
                        help='Latent Dimenstions for encoded outptut',
                        default=const.EPOCHS,
                        type=int
    )
    
    parser.add_argument('--load_from_ckpt_path',
                        help='Load checkpoint weights in model to continue training',
                        default=None,
                        type=str
    )
    
    parser.add_argument('--checkpoint_filepath',
                        help="Path to save model checkpoints while training",
                        default=const.CKPT_PATH,
                        type=str)
    
    parser.add_argument('--verbosity',
                        help="Set Verbosity",
                        choices=[0,1,2],
                        default=const.VERBOSITY,
                        type=int)
    
    args = parser.parse_args()
    
    trainer = Trainer(**vars(args))
    
    trainer.train()
    