import tensorflow as tf

class PSNR_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='PSNR', **kwargs):
        super().__init__(name, **kwargs)
        self.psnr = self.add_variable(
            name='psnr',
            initializer='zeros'
        )
    
    def update_state(self,y_true,y_pred,sample_weight=None):
        psnr = tf.math.reduce_mean(tf.image.psnr(y_true,y_pred,1.0))
        self.psnr.assign(psnr)
        
    def result(self):
        return self.psnr
    
    def reset_state(self):
        self.psnr.assign(0.)
    
class SSIM_Metric(tf.keras.metrics.Metric):
    def __init__(self, name='SSIM', **kwargs):
        super().__init__(name, **kwargs)
        self.ssim = self.add_variable(
            name='ssim',
            initializer='zeros'
        )
        
    def update_state(self, y_true,y_pred,sample_weight=None):
        ssim = tf.math.reduce_mean(tf.image.ssim(y_true,y_pred,1.0))
        self.ssim.assign(ssim)
        
    def result(self):
        return self.ssim
    
    def reset_state(self):
        return self.ssim.assign(0.)