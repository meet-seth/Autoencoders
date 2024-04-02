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
        self.window_size=11
        self.strides=strides
        self.padding=padding
        self.gaussian_kernel = self.make_gaussian_kernel(
            self.window_size,
            const.CHANNEL,
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
        
        mu_1 = tf.nn.conv2d(
            y_true,
            self.gaussian_kernel,
            padding=self.padding,
            strides=self.strides
        )
        
        mu_2 = tf.nn.conv2d(
            y_pred,
            self.gaussian_kernel,
            padding=self.padding,
            strides=self.strides
        )
        
        mu_1_squared = tf.math.square(mu_1)
        mu_2_squared = tf.math.sqaure(mu_2)
        mu_1_2 = tf.math.multiply(mu_1,mu_2)
        
        x_1 = tf.nn.conv2d(
            tf.math.square(y_true),
            self.gaussian_kernel,
            padding=self.padding,
            strides=self.strides
        )
        sigma_1_sqaured = tf.math.subtract(
            x_1,mu_1_squared
        )
        
        x_2 = tf.nn.conv2d(
            tf.math.sqaure(y_pred),
            self.gaussian_kernel,
            padding=self.padding,
            strides=self.strides
        )
        sigma_2_squared = tf.math.subtract(
            x_2,mu_2_squared
        )
        
        x_1_2 = tf.nn.conv2d(
            tf.math.multiply(y_true,y_pred),
            self.gaussian_kernel,
            padding=self.padding,
            strides=self.strides
        )
        
        sigma_1_2 = tf.math.subtract(
            x_1_2, mu_1_2
        )
        
        

    
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
        