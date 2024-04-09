import tensorflow as tf
import argparse
import constants.constants as const
from builders.ModelBuilder import ModelBuilder

class Exporter:
    
    def __init__(self,
                 model,
                 mode,
                 ckpt_path,
                 latent_dims,
                 save_path):
        
        self.model = ModelBuilder(model).build(
            latent_dims=latent_dims,
            log_dir='./log',
            learning_rate=1e-1
        )
        self.model.compile()
        latest = tf.train.latest_checkpoint(ckpt_path)
        self.model.load_weights(latest)
        self.mode = mode
        self.save_path = save_path
        
    def export_model(self):
        
        print("Starting to Export model.")
        
        if self.mode=='full':
            print("Saving in full mode.")
            self.model.save(self.save_path)
            
        elif self.mode=='encoder-decoder':
            print("Saving in Encoder - Decoder Mode.")
            encoder = self.model.encoder
            decoder = self.model.decoder
            
            encoder.save(self.save_path[0])
            decoder.save(self.save_path[1])
            
        print("Model Saved Successfully")
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        help="Path to Model conifg to build model from.",
        required=True
    )
    
    parser.add_argument(
        "--mode",
        help="""One of full or encoder-decoder. 
        full: converts the entire Image Compressor model to saved model or h5 including
        both the encoder and decoder and outptus the regenerated image.
        encoder-decoder: splits the Image Compressor model into encoder part and decoder part, where
        encoder part: converts the input image into latent compressed tensor and
        decoder part: converts latent_dims to output regenerated image.""",
        required=True,
        choices=['full','encoder-decoder']
    )
    
    parser.add_argument(
        "--ckpt_path",
        help="Path to model checkpoints",
        required=True
    )
    
    parser.add_argument(
        "--latent_dims",
        help="Latent Dims to build the latent represent layer from model config",
        default=const.LATENT_DIMS
    )
    
    parser.add_argument(
        "--save_path",
        help="""
        Path to save model. Pass in Directory to save model in Tf Saved Model format.
        Pass in a file path with .h5 extension to save it in h5 format. Pass in a list 
        if type is encoder-decoder.
        """,
        required=True
    )
    
    args = parser.parse_args()
    
    exporter = Exporter(**vars(args))
    exporter.export_model()
    