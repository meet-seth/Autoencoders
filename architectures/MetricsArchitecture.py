import tensorflow as tf

class PSNR(tf.keras.metrics.Metric):
    def __init__(self, name='PSNR', **kwargs):
        super().__init__(name, **kwargs)
        self.psnr = self.add_variable(
            name='psnr',
            initializer='zeros'
        )
    
    def update_state(self,y_pred,y_true):
        mse = tf.keras.losses.mean_squared_error(y_true,y_pred) + .00001
        psnr = tf.math.log(10.) / (-10. * tf.math.log(mse))
        
        self.psnr.assign(psnr)
        
    def result(self):
        return self.psnr
    
class FakeOut(tf.keras.metrics.Metric):
    def __init__(self, name="FakeOut", **kwargs):
        super().__init__(name,**kwargs)
        self.discriminator_loss = self.add_variable(
            name='discriminator_loss',
            initializer='zeros'
        )
        
        self.cross_entropy = tf.keras.losses.binary_crossentropy
        
    def update_state(self, y_true,y_preds):
        disc_loss = self.cross_entropy(y_true,y_preds)     
        self.discriminator_loss.assign(disc_loss)
        
    def result(self):
        return self.discriminator_loss   