import tensorflow as tf

class PSNR_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='PSNR', **kwargs):
        super().__init__(name, **kwargs)
        self.psnr = self.add_variable(
            name='psnr',
            initializer='zeros'
        )
    
    def update_state(self,y_true,y_pred):
        mse = tf.keras.losses.mean_squared_error(y_true,y_pred) + .00001
        psnr = tf.math.log(10.) / (-10. * tf.math.log(mse))
        
        self.psnr.assign(psnr)
        
    def result(self):
        return self.psnr
    
class SSIM_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='SSIM', **kwargs):
        super().__init__(name, **kwargs)
        self.ssim = self.add_variable(
            name='ssim',
            initializer='zeros'
        )
        
    def update_state(self, y_true,y_pred):
        ssim = tf.image.ssim(y_true,y_pred)
        self.ssim.assign(ssim)
        
    def result(self):
        return self.ssim