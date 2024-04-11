import tensorflow as tf
import tensorflow_probability as tfp
import constants.constants as const

    
class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self,weight=const.LOSS_WEIGHT,alpha=const.ALPHA,reduction="auto", name=None):
        super().__init__(reduction, name)
        self.alpha = alpha
        self.weight = weight
    
    def call(self,y_true,y_pred):
        
        ssim_score = 1. - tf.math.reduce_mean(tf.image.ssim(y_true,y_pred,1.0))
        l1_loss = tf.math.reduce_mean(tf.math.abs(y_true-y_pred))
        
        final_loss = self.alpha * ssim_score + (1-self.alpha) * l1_loss
        
        return self.weight * final_loss
        