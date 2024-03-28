import json
import tensorflow as tf
import tensorflow_compression as tfc

class EncoderModel(tf.keras.Model):
    def __init__(self,layers,input_shape,latent_dims,training=True,name='Encoder',**kwargs):
        super().__init__(**kwargs)
        self.latent_dims = latent_dims
        self.name = name
        self.layers = layers
        self.input_shape = input_shape
    def call(self,x):
        x = tf.cast(x,self.compute_dtype) / 255.
        x = tf.reshape(x,self.input_shape)
        for layer in self.layers:
            x = layer(x)
            changes
            
        
        
        compressed = 
class ModelBuilder:
    
    def __init__(self,model_path):
        
        with open(model_path,'r') as f:
            self.model_config = json.load(f)['model']
            f.close()
        
    def build(self,latent_dims):
        model_name = self.model_config['name']
        model_layers = self.model_config['layers']
        
        
        