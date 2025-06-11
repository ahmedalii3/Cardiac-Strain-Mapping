from tensorflow.keras.losses import Loss
import tensorflow as tf

class MaskLoss(Loss):
    """
    Custom Keras loss function that applies a mask to the loss calculation, allowing for selective weighting of regions in the input.
    This loss is designed for tasks where only certain regions of the input (as indicated by a mask) should contribute more heavily to the loss, such as in image segmentation or deformation field estimation.
    Attributes:
        Inherits from the Keras `Loss` class.
    Methods:
        __init__(**kwargs):
            Initializes the MaskLoss instance. Accepts additional keyword arguments for compatibility with Keras configuration.
        call(y_true, y_pred):
            Computes the masked mean squared error loss between `y_true` and `y_pred`.
            Args:
                y_true (tf.Tensor): Ground truth tensor of shape [batch_size, height, width, 3], where the first two channels are the target values and the third channel is the binary mask.
                y_pred (tf.Tensor): Predicted tensor of shape [batch_size, height, width, 2].
            Returns:
                tf.Tensor: Scalar tensor representing the masked mean squared error loss.
            Details:
                - The mask (third channel of y_true) is used to focus the loss on specific regions.
                - The loss is weighted by the ratio of masked to unmasked pixels to balance the contribution of each region.
                - The squared error is computed only for the first two channels.
                - The mask is expanded to match the shape of the squared error for element-wise multiplication.
                - The final loss is the mean of the weighted errors.
        get_config():
            Returns the configuration of the loss instance for serialization.
            Returns:
                dict: Configuration dictionary.
    """
    def __init__(self, **kwargs):  # Add kwargs for config compatibility
        super(MaskLoss, self).__init__(**kwargs)


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
        # weighted_mask = tf.where(mask == 1, final_ratio, 1.0)  # Shape: [batch_size, height, width]

        # Expand dimensions to match squared_error shape
        weighted_mask = tf.expand_dims(mask, axis=-1)  # Shape: [batch_size, height, width, 1]

        # Apply weighted loss
        weighted_error = weighted_mask * squared_error

        mask_sum = tf.reduce_sum(weighted_mask)
        num_of_pixels = tf.cast(128 * 128, tf.float32)
        
        mask_ratio = num_of_pixels / (mask_sum + 1e-6)

        weighted_error = weighted_error * mask_ratio

        mean_error = tf.reduce_mean(weighted_error)

        return mean_error
    
    def get_config(self):  # 🚀 Add this to fix the TypeError!
        base_config = super(MaskLoss, self).get_config()
        return base_config

    
class MAELoss(Loss):
    """
    Mean Absolute Error (MAE) Loss class.
    This loss computes the mean absolute difference between the first two channels of the ground truth tensor (`y_true`) and the predicted tensor (`y_pred`). It is typically used for regression tasks where the goal is to minimize the average absolute error between predictions and targets.
    Methods
    -------
    __init__(**kwargs):
        Initializes the MAELoss instance with optional keyword arguments.
    call(y_true, y_pred):
        Computes the mean absolute error between the first two channels of `y_true` and `y_pred`.
    get_config():
        Returns the configuration of the loss instance for serialization.
    """
    def __init__(self, **kwargs):
        super(MAELoss, self).__init__(**kwargs)


    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true[..., :2]  - y_pred))
    
    def get_config(self):  
        base_config = super(MAELoss, self).get_config()
        return base_config