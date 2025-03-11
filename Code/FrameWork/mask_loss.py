from tensorflow.keras.losses import Loss
import tensorflow as tf

class MaskLoss(Loss):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def call(self, y_true, y_pred):
        # Extract the first two channels (actual ground truth)
        y_true_deformed = y_true[..., :2]  # Shape: [batch_size, height, width, 2]

        # Extract the third channel (mask)
        mask = y_true[..., 2]  # Shape: [batch_size, height, width]

        # Compute the ratio of foreground to background
        num_foreground = tf.reduce_sum(tf.cast(mask == 1, tf.float32))
        num_background = tf.reduce_sum(tf.cast(mask == 0, tf.float32))

        # Compute final_ratio, ensuring stability
        final_ratio = num_background / (num_foreground + 1e-6)
        final_ratio = tf.clip_by_value(final_ratio, 0.1, 10)  # Prevent extreme values

        # Compute squared error
        squared_error = tf.square(y_true_deformed - y_pred)  # Shape: [batch_size, height, width, 2]

        # Assign higher weight to masked region
        weighted_mask = tf.where(mask == 1, final_ratio, 1.0)  # Shape: [batch_size, height, width]

        # Expand dimensions to match squared_error shape
        weighted_mask = tf.expand_dims(weighted_mask, axis=-1)  # Shape: [batch_size, height, width, 1]

        # Apply weighted loss
        weighted_error = weighted_mask * squared_error
        mean_error = tf.reduce_mean(weighted_error)

        return mean_error
    
class MAELoss(Loss):
    def __init__(self):
        super(MAELoss, self).__init__()

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true[..., :2]  - y_pred))