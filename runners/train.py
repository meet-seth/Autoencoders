import tensorflow as tf
from architectures.LossArchitecture import *
from architectures.MetricsArchitecture import *
class Trainer:
    def __init__(self,model,dataset,batch_size,learning_rate,log_dir,epochs):
        self.model = model
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.batch_size =batch_size
        self.log_dir = log_dir
        self.epochs = epochs
        
    def train(self):           
    
        self.model.compile(
            loss = {
                'generator': SSIMLoss()
            },
            metrics = [PSNR_Metric(),SSIM_Metric()]
        )
            
        self.history = self.model.fit(
                self.dataset['train'].batch(self.batch_size).prefetch(tf.data.AUTOTUNE),
                validation_data = self.dataset['val'].batch(self.batch_size).prefetch(tf.data.AUTOTUNE),
                epochs = self.epochs,
                callbacks = [tf.keras.callbacks.TensorBoard(
                    log_dir=self.log_dir,
                    write_graph=True,
                    write_images=True
                )]
            )
        
        return self.model,self.history
        