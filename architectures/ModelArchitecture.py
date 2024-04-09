import tensorflow as tf
class ImageCompressor(tf.keras.Model):
    def __init__(self,inputs,outputs,generator,log_dir,metrics,learning_rate,*args, **kwargs):
        super().__init__(*args,inputs=inputs,outputs=outputs,**kwargs)
        self.custom_metrics = metrics
        self.tf_writer = tf.summary.create_file_writer(log_dir)
        self.generator = generator
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

           
    def call(self,x,training):
        regenerated_output = self.generator(x,training)
        
        return {
            "generator": regenerated_output
        }
    
    @tf.function
    def train_step(self,x):
        
        with tf.GradientTape() as gen_tape:
            predictions = self(x)
            
            true_tensor = {
                "generator": tf.cast(x,self.compute_dtype) / 255.
            }
            loss = {}
            loss['generator_loss'] = self.loss['generator'](true_tensor['generator'],predictions['generator'])
            
        with self.tf_writer.as_default(step=self._train_counter):
            tf.summary.scalar("generator",loss['generator_loss'])
            tf.summary.image("original",x)
            tf.summary.image("regenerated",tf.cast(predictions['generator']*255.,tf.uint8))
            
        generator_gradients = gen_tape.gradient(loss['generator_loss'],self.generator.trainable_variables)
        
        
        self.generator_optimizer.apply_gradients(zip(generator_gradients,self.generator.trainable_variables))

        
        for metric in self.metrics:
            metric.update_state(true_tensor['generator'],predictions['generator'])
     
        mtr = {m.name: m.result() for m in self.metrics}
        mtr['generator_loss'] = loss['generator_loss']
       
        return mtr
    
    @property
    def metrics(self):
        return self.custom_metrics
        
    @tf.function
    def test_step(self,x):
        
        
        
        predictions = self(x)
        true_tensor = {
                "generator": tf.cast(x,self.compute_dtype) / 255.                    
            }
        loss = {}
        loss['generator_loss'] = self.loss['generator'](true_tensor['generator'],predictions['generator'])
        
        with self.tf_writer.as_default(step=self._test_counter):
            tf.summary.scalar("generator",loss['generator_loss'])
            tf.summary.image("original",x)
            tf.summary.image("regenerated",tf.cast(predictions['generator']*255.,tf.uint8))

        
        for metric in self.metrics:
            metric.update_state(true_tensor['generator'],predictions['generator'])
            
        mtr = {m.name: m.result() for m in self.metrics}
        mtr['generator_loss'] = loss['generator_loss']
        
        return mtr
    
# @tf.keras.saving.register_keras_serializable(package="Models",name="Generator")
class Generator(tf.keras.Model):
    def __init__(self,config, pr_input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.pr_input_shape = pr_input_shape
        self._name = self.config['name']
        for sub_model, value in self.config.items():
            if sub_model == 'encoder':
                self.encoder = Encoder(value,pr_input_shape)
            elif sub_model == 'decoder':
                self.decoder = Decoder(value)
                
    def call(self,x,training):
        encoded_output = self.encoder(x)
        regenerated_output = self.decoder(encoded_output)
        
        return regenerated_output
    
    # def get_config(self):
    #     config = super().get_config()
    #     config.update(
    #         {
    #             'config': self.config,
    #             'pr_input_shape': self.pr_input_shape
    #         }
    #     )
    #     return config
    
    # @classmethod
    # def from_config(cls, config, custom_objects=None):
    #     config["config"] = tf.keras.saving.deserialize_keras_object(config["config"])
    #     config["pr_input_shape"] = tf.keras.saving.deserialize_keras_object(config['pr_input_shape'])
        
    #     return cls(**config)
        
                
# @tf.keras.saving.register_keras_serializable(package="Modles",name='Encoder')
class Encoder(tf.keras.Model):
    def __init__(self, sequential_model,pr_input_shape,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.model = sequential_model
        self._name = self.model.name
        self.reshaping = pr_input_shape
        self.dtype_conversion_layer = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32) / 255.)
        self.reshaping_layer = tf.keras.layers.Reshape(target_shape=self.reshaping)
        
    
    @tf.function
    def call(self,x,training):
        x = self.dtype_conversion_layer(x)
        x = self.reshaping_layer(x)
        y = self.model(x)
        
        return y
    
    # def get_config(self):
    #     config = super().get_config()
    #     config.update(
    #         {
    #             'sequential_model': self.model,
    #             'pr_input_shape': self.reshaping
    #         }
    #     )
        
    #     return config
    
    # @classmethod
    # def from_config(cls, config, custom_objects=None):
    #     config['sequential_model'] = tf.keras.saving.deserialize_keras_object(config['sequential_model'])
    #     config['pr_input_shape'] = tf.keras.saving.deserialize_keras_object(config['pr_input_shape'])
        
    #     return cls(**config)
# @tf.keras.saving.register_keras_serializable(package="Layers",name="Clipping")
class ClipplingLayer(tf.keras.layers.Layer):
    def __init__(self,*args,**kwargs):
        super().__init__()
    
    def call(self,inputs):
        return tf.clip_by_value(inputs,clip_value_min=0.,clip_value_max=1.)
    
    # def get_config(self):
    #     return super().get_config()
    
    # @classmethod
    # def from_config(cls, config):
    #     return super().from_config(config)
        
        
# @tf.keras.saving.register_keras_serializable(package="Models",name='Decoder')
class Decoder(tf.keras.Model):
    def __init__(self, sequential_model,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = sequential_model
        self.model.add(ClipplingLayer(name="DecoderClipping"))
        self._name = self.model._name
            
    def call(self,x,training):
        decoder_output = self.model(x)
        
        return decoder_output
    
    # def get_config(self):
    #     config = super().get_config()
    #     config.update(
    #         {
    #             "sequential_model": self.model
    #         }
    #     )
        
    #     return config
    
    # @classmethod
    # def from_config(cls, config, custom_objects=None):
    #     config['sequential_model'] = tf.keras.saving.deserialize_keras_object(config['sequential_model'])
        
    #     return cls(**config)
    