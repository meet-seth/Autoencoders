import tensorflow as tf
import tensorflow_compression as tfc
import constants as const
class ImageCompressor(tf.keras.Model):
    def __init__(self,inputs,outputs,generator,discriminator,log_dir,*args, **kwargs):
        super().__init__(*args,inputs=inputs,outputs=outputs,**kwargs)
        self.tf_writer = tf.summary.create_file_writer(log_dir)
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=const.LEARNING_RATE)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=const.LEARNING_RATE)
           
    def call(self,x,training):
        
        regenerated_output,rate = self.generator(x,training=training)
        discriminator_preds_original = self.discriminator(x,True)
        discriminator_preds_fake = self.discriminator(regenerated_output,False)
        
        return {
            "rate": rate,
            "generator": {
                "image": regenerated_output,
                "fake_out": discriminator_preds_fake
            },
            "discriminator": {
                "real_out": discriminator_preds_original,
                "fake_out": discriminator_preds_fake
            }
        }
        
    
    @tf.function
    def train_step(self,x):
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            predictions = self(x,training=True)
            
            true_tensor = {
                "rate": tf.ones_like(predictions['rate']),
                "generator": {
                    "image": tf.cast(x,self.compute_dtype) / 255.,
                    "fake_out": tf.ones_like(predictions['generator']['fake_out'])
                },
                "discriminator": {
                    "real_out": tf.ones_like(predictions['discriminator']['real_out']),
                    "fake_out": tf.zeros_like(predictions['discriminator']['fake_out'])
                }
            }
            loss = {}
            loss['rate'] = self.loss['rate'](true_tensor['rate'],predictions['rate'])
            loss['generator_loss'] = self.loss['generator'](true_tensor['generator'],predictions['generator'])
            loss['discriminator_loss'] = self.loss['discriminator'](true_tensor['discriminator'],predictions['discriminator'])
            
        with self.tf_writer.as_default(step=self._train_counter):
            tf.summary.scalar("rate",loss['rate'])
            tf.summary.scalar("generator",loss['generator_loss'])
            tf.summary.scalar("discriminator",loss["discriminator_loss"])
            tf.summary.image("original",x)
            tf.summary.image("regenerated",tf.cast(predictions['generator']['image']*255.,tf.uint8))
            
        generator_gradients = gen_tape.gradient(loss['generator_loss'],self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(loss['discriminator_loss'],self.discriminator.trainable_variables)
        
        self.generator_optimizer.apply_gradients(zip(generator_gradients,self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,self.discriminator.trainable_variables))
        
        for metric in self.metrics:
            metric.update_state(loss[metric.name])
        
        return {"generator_loss": loss['generator_loss'],
                "discriminator_loss": loss['discriminator_loss'],
                'rate': loss['rate']}
        
    @tf.function
    def test_step(self,x):
        
        
        
        predictions = self(x,training=False)
        true_tensor = {
                "rate": tf.ones_like(predictions['rate']),
                "generator": {
                    "image": tf.cast(x,self.compute_dtype) / 255.,
                    "fake_out": tf.ones_like(predictions['generator']['fake_out'])
                },
                "discriminator": {
                    "real_out": tf.ones_like(predictions['discriminator']['real_out']),
                    "fake_out": tf.zeros_like(predictions['discriminator']['fake_out'])
                }
            }
        loss = {}
        loss['rate'] = self.loss['rate'](true_tensor['rate'],predictions['rate'])
        loss['generator_loss'] = self.loss['generator'](true_tensor['generator'],predictions['generator'])
        loss['discriminator_loss'] = self.loss['discriminator'](true_tensor['discriminator'],predictions['discriminator'])
        
        with self.tf_writer.as_default(step=self._test_counter):
            tf.summary.scalar("rate",loss['rate'])
            tf.summary.scalar("generator",loss['generator_loss'])
            tf.summary.scalar("discriminator",loss["discriminator_loss"])
            tf.summary.image("original",x)
            tf.summary.image("regenerated",tf.cast(predictions['generator']['image']*255.,tf.uint8))
        
        for metric in self.metrics:
            metric.update_state(loss[metric.name])
        
        return {"generator_loss": loss['generator_loss'],
                "discriminator_loss": loss['discriminator_loss'],
                'rate': loss['rate']}
        
        
    
class Generator(tf.keras.Model):
    def __init__(self,latent_dims,config, input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self._name = self.config['name']
        self.latent_dims = latent_dims
        for sub_model, value in self.config.items():
            if sub_model == 'encoder':
                self.encoder = Encoder(self.latent_dims,value,input_shape)
            elif sub_model == 'decoder':
                self.decoder = Decoder(value)
                
    def call(self,x,training):
        encoded_output, rate = self.encoder(x,training=training)
        regenerated_output = self.decoder(encoded_output)
        
        return regenerated_output, rate
        
                

class Encoder(tf.keras.Model):
    def __init__(self,latent_dims, sequential_model,input_shape,*args, compressed=False,**kwargs):
        super().__init__(*args, **kwargs)
        self.model = sequential_model
        self._name = self.model.name
        self.latent_dims = latent_dims
        self.prior_log_scale= tf.Variable(tf.zeros(self.latent_dims))
        self.compression = compressed
        self.reshaping = input_shape
        self.entropy_model = None
        self.dtype_conversion_layer = tf.keras.layers.Lambda(lambda x: tf.cast(x,self.compute_dtype) / 255.)
        self.reshaping_layer = tf.keras.layers.Reshape(target_shape=self.reshaping)
        
    def set_compression(self,val):
        self.compression = val
        if self.compression:
            self.entropy_model = tfc.ContinuousBatchedEntropyModel(self.prior,coding_rank=1,compression=self.compression)
        else:
            self.entropy_model = None
        
    @property
    def prior(self):
        return tfc.NoisyLogistic(loc=0.,scale = tf.exp(self.prior_log_scale))
    
    @tf.function
    def call(self,x,training):
        x = self.dtype_conversion_layer(x)
        x = self.reshaping_layer(x)
        
        y = self.model(x)
        
        if not self.entropy_model:
            entropy_model = tfc.ContinuousBatchedEntropyModel(self.prior,coding_rank=1,compression=False)
            y_tilde, rank = entropy_model(y,training=training)
            return y_tilde, rank
        else:
            _, bits = self.entropy_model(y,training=False)
            return self.entropy_model.compress(y), bits
            
            

        
class Decoder(tf.keras.Model):
    def __init__(self, sequential_model,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = sequential_model
        self._name = self.model._name
        self.decompression = False
        self.entropy_model = None
        
        
        
    def set_decompression(self,val,entropy_model):
        self.decompression = val
        if self.decompression:
            self.entropy_model = entropy_model
        else:
            self.entropy_model = None
            
    def call(self,x):
        if self.entropy_model:
            x = self.entropy_model.decompress(x)
        decoder_output = self.model(x)
        
        return decoder_output
    
        

class Discriminator(tf.keras.Model):
    def __init__(self, sequential_model,input_shape,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = sequential_model
        self._name = self.model.name
        self.reshaping = input_shape
        self.dtype_conversion_layer = tf.keras.layers.Lambda(lambda x: tf.cast(x,self.compute_dtype) / 255.)
        self.reshaping_layer = tf.keras.layers.Reshape(target_shape=self.reshaping)
    
    @tf.function
    def call(self,x,reshape):
        if reshape:
            x = self.dtype_conversion_layer(x)
            x = self.reshaping_layer(x)
        discriminator_output = self.model(x)
        return discriminator_output
        