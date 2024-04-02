from keras.src.utils.losses_utils import ReductionV2
import tensorflow as tf
import tensorflow_probability as tfp
import constants as const

class RateLoss(tf.keras.losses.Loss):
    def call(self,y_true,y_pred):
        return y_pred
    
class SSIMLoss(tf.keras.losses.Loss):
    def __init__(self, window_size=11,strides=[1,1,1,1],padding='SAME',reduction="auto", name=None):
        super().__init__(reduction, name)
        self.weight = const.SSIM_WEIGHT
        self.window_size=window_size
        self.strides=strides
        self.padding=padding
        self.gaussian_kernel = self.make_gaussian_kernel(
            self.window_size,
            const.CHANNELS,
            sigma=const.SSIM_SIGMA
        )
        self.C1 = tf.constant(0.01 ** 2)
        self.C2 = tf.constant(0.02 ** 2)
        self.M = tf.constant(2.)
    
    def make_gaussian_kernel(self,window_size,channel,mean=0,sigma=1):
            d = tfp.distributions.Normal(mean,sigma)
            size = window_size//2
            val = d.prob(tf.range(start=-size,limit=size+1,delta=1,dtype=tf.float32))
            kernel_2d = tf.einsum('i,j->ij',
                                     val,
                                     val)
            gauss_kernel = kernel_2d / tf.reduce_mean(kernel_2d)
            gauss_kernel = gauss_kernel[:,:,tf.newaxis,tf.newaxis]
            gauss_kernel = tf.tile(gauss_kernel,[1,1,channel,1])
            
            return gauss_kernel
    
    def call(self,y_true,y_pred):
        
        ssim_score = tf.image.ssim(y_true,y_pred,1.0)
        
        return ssim_score
        
        

    
class GeneratorLoss(tf.keras.losses.Loss):
    def __init__(self, reduction='auto',name=None):
        super().__init__(reduction=reduction,name=name)
        self.cross_entropy = tf.keras.losses.binary_crossentropy
    def call(self,y_true,y_pred):
        distortion_loss = tf.math.reduce_mean(abs(y_true['image'] - y_pred['image']))
        discriminator_loss = self.cross_entropy(y_true['fake_out'],y_pred['fake_out'])
        
        total_loss = const.GENERATOR_WEIGHTS['distortion']*distortion_loss + const.GENERATOR_WEIGHTS['discriminator']*discriminator_loss
        
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
        