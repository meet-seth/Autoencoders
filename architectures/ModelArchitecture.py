import tensorflow as tf
class ImageCompressor(tf.keras.Model):
    def __init__(self,inputs,outputs,generator,log_dir,metrics,learning_rate,*args, **kwargs):
        super().__init__(*args,inputs=inputs,outputs=outputs,**kwargs)
        self.custom_metrics = metrics
        self.tf_writer = tf.summary.create_file_writer(log_dir)
        self.generator = generator
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

           
    def call(self,x,training):
        regenerated_output = self.generator(x,training=training)
        
        if not training:
            regenerated_output = self._post_process(x)
        
        return {
            "generator": regenerated_output
        }
        
    @tf.function
    def _post_process(self,x):
        return tf.clip_by_value(x,clip_value_min=0.,clip_value_max=1.)
    
    @tf.function
    def train_step(self,x):
        
        with tf.GradientTape() as gen_tape:
            predictions = self(x,training=True)
            
            true_tensor = {
                "generator": tf.cast(x,self.compute_dtype) / 255.
            }
            loss = {}
            loss['generator_loss'] = self.loss['generator'](true_tensor['generator'],predictions['generator'])
            
        with self.tf_writer.as_default(step=self._train_counter):
            tf.summary.scalar("generator",loss['generator_loss'])
            tf.summary.image("original",x)
            tf.summary.image("regenerated",tf.cast(self._post_process(predictions['generator'])*255.,tf.uint8))
            
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
        
        
        
        predictions = self(x,training=True)
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
    
    
class Generator(tf.keras.Model):
    def __init__(self,config, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self._name = self.config['name']
        for sub_model, value in self.config.items():
            if sub_model == 'encoder':
                self.encoder = Encoder(value,input_shape)
            elif sub_model == 'decoder':
                self.decoder = Decoder(value)
                
    def call(self,x):
        encoded_output = self.encoder(x)
        regenerated_output = self.decoder(encoded_output)
        
        return regenerated_output
        
                

class Encoder(tf.keras.Model):
    def __init__(self, sequential_model,input_shape,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.model = sequential_model
        self._name = self.model.name
        self.reshaping = input_shape
        self.dtype_conversion_layer = tf.keras.layers.Lambda(lambda x: tf.cast(x,self.compute_dtype) / 255.)
        self.reshaping_layer = tf.keras.layers.Reshape(target_shape=self.reshaping)
        
    
    @tf.function
    def call(self,x):
        x = self.dtype_conversion_layer(x)
        x = self.reshaping_layer(x)
        y = self.model(x)
        
        return y
    
class ClipplingLayer(tf.keras.layers.Layer):
    
    def call(self,inputs):
        return tf.clip_by_value(inputs,clip_value_min=0.,clip_value_max=1.)
        
class Decoder(tf.keras.Model):
    def __init__(self, sequential_model,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = sequential_model
        self.model.add(ClipplingLayer())
        self._name = self.model._name
            
    def call(self,x):
        decoder_output = self.model(x)
        
        return decoder_output