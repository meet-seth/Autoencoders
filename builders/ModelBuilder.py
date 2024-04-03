import json
import tensorflow as tf
from architectures.ModelArchitecture import ImageCompressor, Generator
class ModelBuilder:
    
    def __init__(self,model_path):
        
        """
        Build a tf.keras.Model from config.json file
        """
        
        with open(model_path,'r') as f:
            self.model_config = json.load(f)
            f.close()
        
    def build(self,latent_dims,log_dir):
        """
        Starts the model building process. Is called from main.py

        Args:
            latent_dims (int): Number of units in compressed_representation layer

        Returns:
            dict: dictionary with model specifications
        """
        self.latent_dims = latent_dims
        model_conf = self.build_model_from_json(self.model_config)
        
        inputs,outputs,generator = self.generate_outputs(model_conf)
        model = ImageCompressor(inputs=inputs,outputs=outputs,generator=generator,log_dir=log_dir)
        print(model.summary())
        return model
       
    def generate_outputs(self,model_conf):
        name = model_conf['model']['name']
        for sub_model, value in model_conf['model'].items():
            if sub_model =='inputs':
                inputs_layer = value
            elif sub_model == 'generator':
                generator = Generator(value,inputs_layer.shape[1:],name=name)
                
                
        regenerated_output = generator(inputs_layer,training=False)
        outputs_dict = {
            "generator": regenerated_output
        }
        
        return inputs_layer,outputs_dict, generator
        
        
        
    def build_model_from_json(self,config):
        """
        Recursive function that reads json file and accordingly creates either
        a Sequential model or adds the sequential model to dictionary to be further 
        used by tf.keras.Model Functional API

        Args:
            config (dict): Configuration dictionary as read from json

        Returns:
            dict: Dictionary containing tf.keras.Sequential model or layers for
            further use by the tf.keras.Model functional API
        """
        model = {}
        for name, conf in config.items():
            
            if name=='inputs':
                model[name] = self.make_sequential_model(conf,name,inputs=True)
            elif isinstance(conf['layers'],list):
                model[name] = self.make_sequential_model(conf['layers'],name)
            else:
                model[name] = self.build_model_from_json(conf['layers'])
                model[name]['name'] = conf['name']
                
        return model
    
    def make_sequential_model(self,layers,name,inputs=False):
        """
        Generates Sequential Model from the dictionary containing layer specifications

        Args:
            layers (dict): List of Dictionary of Layers
            name (str): name of the model to which the layers belong to
            inputs (bool, optional): Wheather the layers are Input layers or not
            to be processed differently. Defaults to False.

        Returns:
            Either returns a tensorflow layer or tensorflow keras model.    
        """
        
        if not inputs:
            model = tf.keras.Sequential(name=name)
            
        for layer in layers:
            for layer_name, args in layer.items():
                if 'name' in args.keys():
                    if args['name'] == 'compressed_representation':
                        args['units'] = self.latent_dims
                tf_layer = self.get_tf_layer(layer_name,**args)
                
                if not inputs:
                    model.add(tf_layer)
                else:
                    return tf_layer
        return model
                
    def get_tf_layer(self,layer_name,**kwargs):
        """
        Get tf.keras.layer by passing in layer_name in string 

        Args:
            layer_name (str): Type of Layer, Conv2D, Dense, Flatten, etc.

        Returns:
            tf.keras.Layer: A tensorflow layer
        """
        return getattr(tf.keras.layers,layer_name)(**kwargs)