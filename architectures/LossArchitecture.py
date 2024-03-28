from keras.src.utils.losses_utils import ReductionV2
import tensorflow as tf

class RateLoss(tf.keras.losses.Loss):
    def call(self,y_true,y_pred):
        return y_pred
    
class GeneratorLoss(tf.keras.losses.Loss):
    def __init__(self, reduction='auto',name=None):
        super().__init__(reduction=reduction,name=name)
        self.cross_entropy = tf.keras.losses.binary_crossentropy
    def call(self,y_true,y_pred):
        print(y_true)
        print(y_pred)
        distortion_loss = tf.math.reduce_mean(abs(y_true['image'] - y_pred['image']))
        discriminator_loss = self.cross_entropy(y_true['fake_out'],y_pred['fake_out'])
        
        total_loss = discriminator_loss + distortion_loss
        
        return total_loss
    
class DiscriminatorLoss(tf.keras.losses.Loss):
    def __init__(self, reduction='auto',name=None):
        super().__init__(reduction=reduction,name=name)
        self.cross_entropy = tf.keras.losses.binary_crossentropy
        
    def call(self,y_true,y_preds):

        real_loss = self.cross_entropy(y_true['real_out'],y_preds['real_out'])
        fake_loss = self.cross_entropy(y_true['fake_out'],y_preds['fake_out'])
        
        total_loss = real_loss + fake_loss
        
        return total_loss
        