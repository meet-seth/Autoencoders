import tensorflow as tf
import tensorflow_probability as tfp
import constants.constants as const

class RateLoss(tf.keras.losses.Loss):
    def call(self,y_true,y_pred):
        return y_pred
    
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
        
        

    
class GeneratorLoss(tf.keras.losses.Loss):
    def __init__(self, reduction='auto',name=None):
        super().__init__(reduction=reduction,name=name)
        self.cross_entropy = tf.keras.losses.binary_crossentropy
    def call(self,y_true,y_pred):
        distortion_loss = tf.math.reduce_mean(abs(y_true['image'] - y_pred['image']))
        discriminator_loss = self.cross_entropy(y_true['fake_out'],y_pred['fake_out'])
        
        total_loss = const.GENERATOR_WEIGHTS['distortion']*distortion_loss + const.GENERATOR_WEIGHTS['discriminator']*discriminator_loss
        
        return total_loss
        