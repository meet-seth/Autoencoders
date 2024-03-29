import tensorflow as tf
class Validator:
    def __init__(self,model,dataset,batch_size,log_dir):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.log_dir = log_dir
        
    def evalulate(self):
            losses_dict = self.model.evaluate(
                self.dataset,
                batch_size=self.batch_size,
                callbacks=[tf.keras.callbacks.TensorBoard(
                    log_dir=self.log_dir,
                    write_graph=True,
                    write_images=True,
                )],
                return_dict=True
            )
            
            return losses_dict