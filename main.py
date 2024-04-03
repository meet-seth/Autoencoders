import argparse
import constants as const
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('INFO')
from builders.ModelBuilder import ModelBuilder
from builders.DatasetBuilder import DatasetBuilder
from runners.train import Trainer
from runners.validation import Validator
from runners.inference import Inference

class Process:
    
    def __init__(self,
                 mode,
                 model,
                 tfds,
                 dataset,
                 log_dir,
                 batch_size,
                 learning_rate,
                 latent_dims,
                 epochs,
                 save_dir,
                 model_export_dir,
                 model_name):
        
        const.LEARNING_RATE = learning_rate        
        if model.endswith('.json'):
            self.model = ModelBuilder(model).build(latent_dims,log_dir)
        else:
            self.model = tf.keras.models.load_model(model)
        self.mode = mode
        self.dataset = DatasetBuilder(tfds,dataset).build()
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.histories = []
        self.save_dir = save_dir
        self.model_export_dir = model_export_dir
        self.model_name = model_name
    
    def start(self):
        if self.mode=='train':
            self.run_training()
        
        elif self.mode=='val':
            self.run_validation()
            
        elif self.mode=='pred':
            self.run_inference()
        
    def run_training(self):
        trainer = Trainer(model=self.model,
                          dataset=self.dataset,
                          batch_size=self.batch_size,
                          learning_rate=self.learning_rate,
                          log_dir=self.log_dir,
                          epochs=self.epochs)
        self.model, self.histories = trainer.train()
        self.save_model(self.model)
    
    def run_validation(self):
        validator = Validator(
            model=self.model,
            dataset=self.dataset['val'],
            batch_size=self.batch_size,
            log_dir=self.log_dir
        )

        self.losses_dict = validator.evalulate()
        
    def run_inference(self):
        
        predictor = Inference(
            self.model,
            self.save_dir
        )
        
        self.inferred_outputs = predictor.infer(self.dataset)
    
    def save_model(self,model):
        
        print(f"Saving model at {self.model_export_dir}/{self.model_name}.keras")
        model.save(f"{self.model_export_dir}/{self.model_name}.keras")
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode",
                        default="train",
                        help="Define train validation or prediction mode.", 
                        choices=['train','val','pred'],
                        type=str,
                        required=True
    )
    
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
    parser.add_argument("--save_dir",
                        help="Saving Directory for infered images",
                        default=None,
                        type=str
    )
    parser.add_argument('--model_export_dir',
                        help="Save model to this directory",
                        default="./model",
                        type=str
    )
    parser.add_argument('--model_name',
                        help="Name of the model file to be exported to",
                        default="ImageCompressor",
                        type=str)
    
    parser.add_argument # Add Verbosity Argument
    
    args = parser.parse_args()
    
    process = Process(**vars(args))
    
    process.start()