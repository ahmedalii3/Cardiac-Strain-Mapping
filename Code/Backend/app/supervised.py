import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model

# Custom Loss Classes
class MaskLoss(Loss):
    def __init__(self, **kwargs):
        super(MaskLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_true_deformed = y_true[..., :2]
        mask = y_true[..., 2]
        num_foreground = tf.reduce_sum(tf.cast(mask == 1, tf.float32))
        num_background = tf.reduce_sum(tf.cast(mask == 0, tf.float32))
        final_ratio = num_background / (num_foreground + 1e-6)
        final_ratio = tf.clip_by_value(final_ratio, 0.1, 10)
        squared_error = tf.square(y_true_deformed - y_pred)
        weighted_mask = tf.expand_dims(mask, axis=-1)
        weighted_error = weighted_mask * squared_error
        mask_sum = tf.reduce_sum(weighted_mask)
        num_of_pixels = tf.cast(128 * 128, tf.float32)
        mask_ratio = num_of_pixels / (mask_sum + 1e-6)
        weighted_error = weighted_error * mask_ratio
        mean_error = tf.reduce_mean(weighted_error)
        return mean_error

    def get_config(self):
        return super(MaskLoss, self).get_config()

class MAELoss(Loss):
    def __init__(self, **kwargs):
        super(MAELoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true[..., :2] - y_pred))

    def get_config(self):
        return super(MAELoss, self).get_config()

# Custom Model Classes
class Conv_block(tf.keras.Model):
    def __init__(self, num_filters):
        super(Conv_block, self).__init__()
        self.num_filters = num_filters
        self.conv1 = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.conv2 = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def get_config(self):
        config = super(Conv_block, self).get_config()
        config.update({'num_filters': self.num_filters})
        return config

class UpConv_block(tf.keras.Model):
    def __init__(self, num_filters):
        super(UpConv_block, self).__init__()
        self.num_filters = num_filters
        self.upconv = Conv2DTranspose(num_filters, 3, strides=2, padding='same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')

    def call(self, inputs):
        x = self.upconv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x

    def get_config(self):
        config = super(UpConv_block, self).get_config()
        config.update({'num_filters': self.num_filters})
        return config

class Max_pool(tf.keras.Model):
    def __init__(self):
        super(Max_pool, self).__init__()
        self.pool = MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):
        x = self.pool(inputs)
        return x

    def get_config(self):
        return super(Max_pool, self).get_config()

class Unet(tf.keras.Model):
    def __init__(self, trainable=True, dtype=None, **kwargs):
        super(Unet, self).__init__(**kwargs)
        self.conv_block1 = Conv_block(64)
        self.pool1 = Max_pool()
        self.conv_block2 = Conv_block(128)
        self.pool2 = Max_pool()
        self.conv_block3 = Conv_block(256)
        self.pool3 = Max_pool()
        self.conv_block4 = Conv_block(512)
        self.pool4 = Max_pool()
        self.conv_block5 = Conv_block(1024)
        self.upconv_block1 = UpConv_block(512)
        self.conv_block6 = Conv_block(512)
        self.upconv_block2 = UpConv_block(256)
        self.conv_block7 = Conv_block(256)
        self.upconv_block3 = UpConv_block(128)
        self.conv_block8 = Conv_block(128)
        self.upconv_block4 = UpConv_block(64)
        self.conv_block9 = Conv_block(64)
        self.output_def = Conv2D(2, 1, activation='linear')

    def call(self, inputs):
        moving, fixed = inputs
        inputs = Concatenate()([moving, fixed])
        conv1 = self.conv_block1(inputs)
        pool1 = self.pool1(conv1)
        conv2 = self.conv_block2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv_block3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv_block4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv_block5(pool4)
        upconv1 = self.upconv_block1(conv5)
        concat1 = Concatenate()([conv4, upconv1])
        conv6 = self.conv_block6(concat1)
        upconv2 = self.upconv_block2(conv6)
        concat2 = Concatenate()([conv3, upconv2])
        conv7 = self.conv_block7(concat2)
        upconv3 = self.upconv_block3(conv7)
        concat3 = Concatenate()([conv2, upconv3])
        conv8 = self.conv_block8(concat3)
        upconv4 = self.upconv_block4(conv8)
        concat4 = Concatenate()([conv1, upconv4])
        conv9 = self.conv_block9(concat4)
        output = self.output_def(conv9)
        return output

    def get_config(self):
        return super(Unet, self).get_config()