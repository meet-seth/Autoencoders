import tensorflow as tf
class Inference:
    def __init__(self,model,save_dir=None):
        self.counter = 0
        self.model = model
        self.save_dir = save_dir        
        
    def infer(self,data,post_process=True):
        training = not post_process
        outputs = self.model(data,training=training)
        
        if self.save_dir is not None:
            
            for i in outputs:
                tf.keras.utils.save_img(
                    f'{self.save_dir}/{self.counter}.jpg',
                    i.numpy(),
                    data_format='channels_last',
                    scale=True,
                )
                
                self.counter += 1
        
        return outputs
                    
            
        
            
    