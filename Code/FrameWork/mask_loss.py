from tensorflow.keras.losses import Loss
import tensorflow as tf

class Mask_loss(Loss):
    def __init__(self):
        super(Mask_loss, self).__init__()
        
    def call(self, y_true, y_pred):
        mask = y_true[..., 2]
        y_true_deformed = y_true[..., :2]
        error = y_true_deformed - y_pred
        squared_error = error**2
        masked_error = mask * squared_error
        mean_error = tf.reduce_mean(masked_error)
        return mean_error
