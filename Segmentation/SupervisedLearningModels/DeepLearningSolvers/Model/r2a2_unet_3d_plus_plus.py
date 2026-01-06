from re import I
import os
from utils import utils
import tensorflow as tf
from tensorflow import reduce_sum
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Reshape, concatenate, Conv2D, MaxPooling3D, UpSampling3D, BatchNormalization, Add, Activation, Dropout, Flatten, Conv3DTranspose, Layer, Dense, GlobalAveragePooling3D, GlobalMaxPooling3D, Multiply, Lambda, AveragePooling3D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.ndimage import distance_transform_edt
import numpy as np
import cv2
        
class channel_attention(Layer):
    """ 
    channel attention module 
    
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    def __init__(self, ratio=8, **kwargs):
        super(channel_attention, self).__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super(channel_attention, self).get_config()
        config.update({'ratio': self.ratio})
        return config

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_layer_one = Dense(
            channel // self.ratio,
            activation='elu',
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        self.shared_layer_two = Dense(
            channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros'
        )
        super(channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        channel = inputs.get_shape().as_list()[-1]

        avg_pool = GlobalAveragePooling3D()(inputs)
        avg_pool = Reshape((1, 1, 1, channel))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = GlobalMaxPooling3D()(inputs)
        max_pool = Reshape((1, 1, 1, channel))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        feature = Add()([avg_pool, max_pool])
        feature = Activation('sigmoid')(feature)

        return Multiply()([inputs, feature])

class spatial_attention(Layer):
    """ spatial attention module 
        
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    def __init__(self, kernel_size=7, **kwargs):
        super(spatial_attention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def get_config(self):
        config = super(spatial_attention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

    def build(self, input_shape):
        self.conv3d = Conv3D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False
        )
        super(spatial_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = concatenate([avg_pool, max_pool], axis=-1)
        feature = self.conv3d(concat)

        return Multiply()([inputs, feature])

class Channel_attention(tf.keras.layers.Layer):
    """ 
    Channel attention module 
    
    Fu, Jun, et al. "Dual attention network for scene segmentation." 
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(
        self,
        gamma_initializer=tf.zeros_initializer(),
        gamma_regularizer=None,
        gamma_constraint=None,
        **kwargs
    ):
        super(Channel_attention, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def get_config(self):
        config = super(Channel_attention, self).get_config()
        config.update(
            {
                'gamma_initializer': self.gamma_initializer,
                'gamma_regularizer': self.gamma_regularizer,
                'gamma_constraint': self.gamma_constraint
            }
        )
        return config

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1, ),
            initializer=self.gamma_initializer,
            name='gamma',
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint
        )
        super(Channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape(
            (input_shape[1] * input_shape[2] * input_shape[3], input_shape[4])
        )(inputs)
        proj_key = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        energy = tf.keras.backend.batch_dot(proj_query, proj_key)
        attention = tf.keras.activations.softmax(energy)

        outputs = tf.keras.backend.batch_dot(attention, proj_query)
        outputs = tf.keras.layers.Reshape(
            (input_shape[1], input_shape[2], input_shape[3], input_shape[4])
        )(outputs)
        outputs = self.gamma * outputs + inputs

        return outputs

class Position_attention(tf.keras.layers.Layer):
    """ 
    Position attention module 
        
    Fu, Jun, et al. "Dual attention network for scene segmentation." 
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(
        self,
        ratio=8,
        gamma_initializer=tf.zeros_initializer(),
        gamma_regularizer=None,
        gamma_constraint=None,
        **kwargs
    ):
        super(Position_attention, self).__init__(**kwargs)
        self.ratio = ratio
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def get_config(self):
        config = super(Position_attention, self).get_config()
        config.update(
            {
                'ratio': self.ratio,
                'gamma_initializer': self.gamma_initializer,
                'gamma_regularizer': self.gamma_regularizer,
                'gamma_constraint': self.gamma_constraint
            }
        )
        return config

    def build(self, input_shape):
        super(Position_attention, self).build(input_shape)
        self.query_conv = tf.keras.layers.Conv3D(
            filters=input_shape[-1] // self.ratio,
            kernel_size=(1, 1, 1),
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.key_conv = tf.keras.layers.Conv3D(
            filters=input_shape[-1] // self.ratio,
            kernel_size=(1, 1, 1),
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.value_conv = tf.keras.layers.Conv3D(
            filters=input_shape[-1],
            kernel_size=(1, 1, 1),
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.gamma = self.add_weight(
            shape=(1, ),
            initializer=self.gamma_initializer,
            name='gamma',
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape(
            (input_shape[1] * input_shape[2] * input_shape[3], input_shape[4] // self.ratio)
        )(self.query_conv(inputs))
        proj_query = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        proj_key = tf.keras.layers.Reshape(
            (input_shape[1] * input_shape[2] * input_shape[3], input_shape[4] // self.ratio)
        )(self.key_conv(inputs))
        energy = tf.keras.backend.batch_dot(proj_key, proj_query)
        attention = tf.keras.activations.softmax(energy)

        proj_value = tf.keras.layers.Reshape(
            (input_shape[1] * input_shape[2] * input_shape[3], input_shape[4])
        )(self.value_conv(inputs))

        outputs = tf.keras.backend.batch_dot(attention, proj_value)
        outputs = tf.keras.layers.Reshape(
            (input_shape[1], input_shape[2], input_shape[3], input_shape[4])
        )(outputs)
        outputs = self.gamma * outputs + inputs

        return outputs
        
class R2A2_UNet_plus_plus:
  """
  Class for Deep Learning Hyperspectral Segmentation with the newly proposed 3D-R2-Atrous-CBAM-Focus-UNet++
  Input:  utils: the utilility class
  """
  def __init__(self, utils):
    self.utils = utils
    self.data_path = os.path.join(os.getcwd(), '')
  
  def cbam_block(self, feature, ratio=16, kernel_size=7):
    """
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    feature = channel_attention(ratio=ratio)(feature)
    feature = spatial_attention(kernel_size=kernel_size)(feature)
    return feature
    
  def encoding_layers_building_blocks(self, units, in_layer, downsampling=True):
      '''
      This method returns the encoding layers building blocks (i.e, downsampling layers)
      '''
      input_shortcut = Conv3D(units, (1, 1, 1), padding='same', kernel_initializer='he_normal')(in_layer)
      convolution_layer_main = input_shortcut
      
      # Loop through each convolutional stacks and recurrent layers
      for i in range(self.utils.r2unet_stacks):
        internal_residual_layer = BatchNormalization()(convolution_layer_main)
        internal_residual_layer = Activation('elu')(internal_residual_layer)
        internal_residual_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(internal_residual_layer)
        for j in range(self.utils.r2unet_recur_num):
            add_layer = Add()([internal_residual_layer, convolution_layer_main])
            internal_residual_layer = BatchNormalization()(add_layer)
            internal_residual_layer = Activation('elu')(internal_residual_layer)
            internal_residual_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(internal_residual_layer)
        convolution_layer_main = internal_residual_layer
      out_layer_before_pooling = Add()([convolution_layer_main, input_shortcut])
      out_layer_before_pooling = self.cbam_block(out_layer_before_pooling)
      if downsampling == True:
          out_layer_1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(out_layer_before_pooling)
          out_layer_2 = AveragePooling3D(pool_size=(2, 2, 2), padding='same')(out_layer_before_pooling)
          out_layer = concatenate([out_layer_1, out_layer_2], axis=4)
          return out_layer, out_layer_before_pooling
      else:
          return out_layer_before_pooling
    
  def decoding_layers_building_blocks(self, units, in_layer, concat_layer):
      '''
      This method returns the decoding layers building blocks (i.e, upsampling layers)
      '''
      if concat_layer.shape[3] == 1:
          upsampling_layer = UpSampling3D(size=(2, 2, 1))(in_layer)
      else:
          upsampling_layer = UpSampling3D(size=(2, 2, 2))(in_layer)

      upsampling_layer = concatenate([upsampling_layer, concat_layer], axis=4)
      
      input_shortcut = Conv3D(units, (1, 1, 1), padding='same', kernel_initializer='he_normal')(upsampling_layer)
      convolution_layer_main = input_shortcut
      
      # Loop through each convolutional stacks and recurrent layers
      for i in range(self.utils.r2unet_stacks):
        internal_residual_layer = BatchNormalization()(convolution_layer_main)
        internal_residual_layer = Activation('elu')(internal_residual_layer)
        internal_residual_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(internal_residual_layer)
        for j in range(self.utils.r2unet_recur_num):
            add_layer = Add()([internal_residual_layer, convolution_layer_main])
            internal_residual_layer = BatchNormalization()(add_layer)
            internal_residual_layer = Activation('elu')(internal_residual_layer)
            internal_residual_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(internal_residual_layer)
        convolution_layer_main = internal_residual_layer
      out_layer = Add()([convolution_layer_main, input_shortcut])
      out_layer = self.cbam_block(out_layer)
      return out_layer
  
  def focus_gate_building_blocks(self, units, x, g):
      '''
      This method returns the focus layer in the 2021 paper, however, the paper has made a small mistake that upsamples twice the input, we corrected it here.
      '''
      shape_x = K.int_shape(x)
      shape_g = K.int_shape(g)
      
      # Getting the gating signal to the same number of filters as the inter_shape
      phi_g = Conv3D(units, (1, 1, 1), padding='same', kernel_initializer='he_normal')(g)

      # Getting the x signal to the same shape as the gating signal
      theta_x = Conv3D(units, (1, 1, 1), strides=(shape_x[1]//shape_g[1], shape_x[2]//shape_g[2], shape_x[3]//shape_g[3]), padding='same', kernel_initializer='he_normal')(x)

      # Element-wise addition of the gating and x signals
      add_xg = Add()([phi_g, theta_x])
      add_xg = Activation('elu')(add_xg)

      # channel attention
      channel_attention_ = Channel_attention()(add_xg)

      # spatial attention
      spatial_attention_ = Position_attention()(add_xg)
      
      # combine channel and spatial weights
      weights = Multiply()([channel_attention_, spatial_attention_])
      
      shape_weight = K.int_shape(weights)
      # Upsampling psi back to the original dimensions of x signal
      upsample_sigmoid_xg = Conv3DTranspose(units, (shape_x[1] // shape_weight[1], shape_x[2] // shape_weight[2], shape_x[3] // shape_weight[3]), strides=(shape_x[1] // shape_g[1], shape_x[2] // shape_g[2], shape_x[3] // shape_g[3]), padding='same', kernel_initializer='he_normal')(weights)

      # Element-wise multiplication of attention coefficients back onto original x signal
      focus_coefficients = Multiply()([upsample_sigmoid_xg, x])

      # Final 1x1x1 convolution to consolidate attention signal to original x dimensions
      output = Conv3D(shape_x[4], (1, 1, 1), strides=(1, 1, 1), padding='same', kernel_initializer='he_normal')(focus_coefficients)
      out_layer = BatchNormalization()(output)
      return out_layer
      
  def dice_loss(self, y_true, y_pred, smooth=1e-6):
    # Calculate intersection and the sum for the numerator and denominator of the Dice score
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_true_pred = K.sum(y_true, axis=[0, 1, 2]) + K.sum(y_pred, axis=[0, 1, 2])

    # Calculate the Dice score for each class
    dice_scores = (2. * intersection + smooth) / (sum_true_pred + smooth)
    dice_loss_multiclass = 1 - K.mean(dice_scores)
    return dice_loss_multiclass

  def compute_sdm(self, y_true_class):
    """
    Compute the Signed Distance Map (SDM) for a single class in a one-hot encoded ground truth.
    Positive values inside the boundary, negative values outside.
    """
    # Compute the signed distance map using scipy's distance transform
    sdm_inside = distance_transform_edt(y_true_class)
    sdm_outside = distance_transform_edt(1 - y_true_class)
    return sdm_inside - sdm_outside

  def boundary_loss(self, y_true, y_pred, smooth=1e-6):
    """
    Boundary loss function using Signed Distance Maps (SDM).
    
    Args:
        y_true: Ground truth one-hot encoded mask (shape: [batch_size, 256, 256, 9]).
        y_pred: Predicted mask (shape: [batch_size, 256, 256, 9]).
        smooth: Smoothing factor to avoid division by zero.
    
    Returns:
        Boundary loss value.
    """
    # Get the number of classes
    num_classes = K.shape(y_true)[-1]
    
    # Initialize the total loss
    total_loss = K.cast(0.0, dtype='float32')
    
    for class_id in range(num_classes):
        # Extract the binary mask for the current class from y_true and y_pred
        y_true_class = y_true[..., class_id]
        y_pred_class = y_pred[..., class_id]
        
        # Compute the SDM for the current class in the ground truth
        sdm_class = tf.py_function(func=self.compute_sdm, inp=[y_true_class], Tout=tf.float32)
        
        # Calculate the boundary loss as the mean absolute error between the SDM and the predicted mask
        boundary_loss_class = K.mean(K.abs(sdm_class * (y_pred_class - y_true_class)))
        
        # Accumulate the loss for each class
        total_loss += boundary_loss_class
    
    # Return the mean loss over all classes
    return total_loss / K.cast(num_classes, dtype='float32')
    
  def tversky_loss(self, y_true, y_pred, alpha=0.5, beta=0.5, gamma=1.33, smooth=1e-6):
    # Focal + Tversky: Calculate true positives (intersection), false negatives, and false positives
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    false_negatives = K.sum(y_true * (1 - y_pred), axis=[0, 1, 2])
    false_positives = K.sum((1 - y_true) * y_pred, axis=[0, 1, 2])

    # Tversky index formula
    tversky_index = (intersection + smooth) / (intersection + alpha * false_positives + beta * false_negatives + smooth)

    focal_tversky = K.pow((1 - tversky_index), gamma)
    ft = K.mean(focal_tversky)
    boundary_loss = self.boundary_loss(y_true, y_pred)
    return 0.5*ft + 0.5*boundary_loss
      
  def build_3d_r2unet_plus_plus(self):
    '''
    This function is a tensorflow realization of the 3D-R2A2-Focus-U-Net++ Model (ours). 
    '''
    input_layer = Input((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep, 1))
    
    # L1
    down_sampling_pooling_layer_1, encoding_space_layer_1 = self.encoding_layers_building_blocks(64, input_layer)
    down_sampling_pooling_layer_2, encoding_space_layer_2 = self.encoding_layers_building_blocks(128, down_sampling_pooling_layer_1)
    up_output_focus_layer_1 = self.focus_gate_building_blocks(64, encoding_space_layer_1, encoding_space_layer_2)
    up_output_1 = self.decoding_layers_building_blocks(64, encoding_space_layer_2, up_output_focus_layer_1)
    
    # L2
    down_sampling_pooling_layer_3, encoding_space_layer_3 = self.encoding_layers_building_blocks(256, down_sampling_pooling_layer_2)
    up_output_focus_layer_22 = self.focus_gate_building_blocks(128, encoding_space_layer_2, encoding_space_layer_3)
    up_output_22 = self.decoding_layers_building_blocks(128, encoding_space_layer_3, up_output_focus_layer_22)
    up_output_focus_layer_21 = self.focus_gate_building_blocks(64, up_output_1, encoding_space_layer_3)
    concat_layer_21 = concatenate([up_output_focus_layer_1, up_output_focus_layer_21], axis=4)
    up_output_2 = self.decoding_layers_building_blocks(64, up_output_22, concat_layer_21)

    # L3
    down_sampling_pooling_layer_4, encoding_space_layer_4 = self.encoding_layers_building_blocks(512, down_sampling_pooling_layer_3)
    up_output_focus_layer_33 = self.focus_gate_building_blocks(256, encoding_space_layer_3, encoding_space_layer_4)
    up_output_33 = self.decoding_layers_building_blocks(256, encoding_space_layer_4, up_output_focus_layer_33)
    up_output_focus_layer_32 = self.focus_gate_building_blocks(128, up_output_22, encoding_space_layer_4)
    concat_layer_32 = concatenate([up_output_focus_layer_22, up_output_focus_layer_32], axis=4)
    up_output_32 = self.decoding_layers_building_blocks(128, up_output_33, concat_layer_32)
    up_output_focus_layer_31 = self.focus_gate_building_blocks(64, up_output_2, encoding_space_layer_4)
    concat_layer_31 = concatenate([up_output_focus_layer_1, up_output_focus_layer_21, up_output_focus_layer_31], axis=4)
    up_output_3 = self.decoding_layers_building_blocks(64, up_output_32, concat_layer_31)
    
    # L4
    encoding_space_output_layer = self.encoding_layers_building_blocks(1024, down_sampling_pooling_layer_4, downsampling=False)
    up_output_focus_layer_44 = self.focus_gate_building_blocks(512, encoding_space_layer_4, encoding_space_output_layer)
    up_output_44 = self.decoding_layers_building_blocks(512, encoding_space_output_layer, up_output_focus_layer_44)
    up_output_focus_layer_43 = self.focus_gate_building_blocks(256, up_output_33, encoding_space_output_layer)
    concat_layer_43 = concatenate([up_output_focus_layer_33, up_output_focus_layer_43], axis=4)
    up_output_43 = self.decoding_layers_building_blocks(256, up_output_44, concat_layer_43)
    up_output_focus_layer_42 = self.focus_gate_building_blocks(128, up_output_32, encoding_space_output_layer)
    concat_layer_42 = concatenate([up_output_focus_layer_22, up_output_focus_layer_32, up_output_focus_layer_42], axis=4)
    up_output_42 = self.decoding_layers_building_blocks(128, up_output_43, concat_layer_42)
    up_output_focus_layer_41 = self.focus_gate_building_blocks(64, up_output_3, encoding_space_output_layer)
    concat_layer_41 = concatenate([up_output_focus_layer_1, up_output_focus_layer_21, up_output_focus_layer_31, up_output_focus_layer_41], axis=4)
    up_output_4 = self.decoding_layers_building_blocks(64, up_output_42, concat_layer_41)
    
    
    # classification blocks
    output_layer_1 = Conv3D(self.utils.n_features, (1, 1, self.utils.num_components_to_keep))(up_output_1)
    output_layer_1 = Reshape((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.n_features))(output_layer_1)
    output_layer_1 = Activation('softmax', name='output_layer_1')(output_layer_1)
    
    output_layer_2 = Conv3D(self.utils.n_features, (1, 1, self.utils.num_components_to_keep))(up_output_2)
    output_layer_2 = Reshape((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.n_features))(output_layer_2)
    output_layer_2 = Activation('softmax', name='output_layer_2')(output_layer_2)
    
    output_layer_3 = Conv3D(self.utils.n_features, (1, 1, self.utils.num_components_to_keep))(up_output_3)
    output_layer_3 = Reshape((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.n_features))(output_layer_3)
    output_layer_3 = Activation('softmax', name='output_layer_3')(output_layer_3)
    
    output_layer_4 = Conv3D(self.utils.n_features, (1, 1, self.utils.num_components_to_keep))(up_output_4)
    output_layer_4 = Reshape((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.n_features))(output_layer_4)
    output_layer_4 = Activation('softmax', name='output_layer_4')(output_layer_4)
    
    if self.utils.deep_supervision == True:
        model = Model(inputs=[input_layer], outputs=[output_layer_1, output_layer_2, output_layer_3, output_layer_4])
        learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer_1': self.tversky_loss, 'output_layer_2': self.tversky_loss, 'output_layer_3': self.tversky_loss, 'output_layer_4': self.tversky_loss}, metrics={'output_layer_1': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_2': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_3': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_4': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)})
    else:
        model = Model(inputs=[input_layer], outputs=[output_layer_4])
        learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer_4': self.tversky_loss}, metrics={'output_layer_4': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)})
    return model

  def train(self):
    '''
    This method internally train the U-Net defined above, with a given set of hyperparameters.
    '''
    if self.utils.pre_load_dataset == True:
      print("Loading Training & Validation Dataset...")
      self.utils.X_train = np.load(os.path.join(self.data_path, 'Data/X_train.npy')).astype(np.float64)
      self.utils.X_validation = np.load(os.path.join(self.data_path, 'Data/X_validation.npy')).astype(np.float64)
      self.utils.y_train = np.load(os.path.join(self.data_path, 'Data/y_train.npy')).astype(np.float64)
      self.utils.y_validation = np.load(os.path.join(self.data_path, 'Data/y_validation.npy')).astype(np.float64)
      print("Training and Validation Dataset Loaded")
    else:
      print("Data Loading...")
      X, y = self.utils.DataLoader(self.utils.dataset)
      print("Data Loading Completed")
      if self.utils.svd_denoising == True:
        print("Raw Data Denoising...")
        denoised_data, variance_matrix = self.utils.svd_denoise(X, n_svd = self.utils.n_svd)
        # Compute cumulative variance explained
        cumulative_variance = np.cumsum(variance_matrix**2) / np.sum(variance_matrix**2)
        while True:
          if cumulative_variance[self.utils.n_svd - 1] <= self.utils.svd_denoise_threshold:
              self.utils.n_svd += 1
              denoised_data, variance_matrix = self.utils.svd_denoise(X, n_svd = self.utils.n_svd)
              cumulative_variance = np.cumsum(variance_matrix**2) / np.sum(variance_matrix**2)
          else:
              X = denoised_data
              break
        print("Raw Data Denoise Completed")
      else:
        pass
      if self.utils.layer_standardization == True:
        print("Layer-Wise Standardization...")
        X_normalized = self.utils.run_layer_standardization(X)
        print("Layer-Wise Standardization Completed")
      else:
        print("Layer-Wise Normalization...")
        X_normalized = self.utils.run_layer_normalization(X)
        print("Layer-Wise Normalization Completed")
      print("Prepare Data for Training, Validation & Testing...")
      X_processed, y_processed = self.utils.prepare_dataset_for_training(X_normalized, y)
      X_train, X_, y_train, y_ = train_test_split(X_processed, y_processed, train_size = 1 - self.utils.test_ratio, test_size = self.utils.test_ratio, random_state=1234)
      X_validation, X_test, y_validation, y_test = train_test_split(X_, y_, train_size = 1 - self.utils.test_ratio, test_size = self.utils.test_ratio, random_state=4321)
      self.utils.X_train, self.utils.X_validation, self.utils.X_test = X_train, X_validation, X_test
      self.utils.y_train, self.utils.y_validation, self.utils.y_test = y_train, y_validation, y_test
      np.save(os.path.join(self.data_path, 'Data/X_train.npy'), self.utils.X_train)
      np.save(os.path.join(self.data_path, 'Data/X_validation.npy'), self.utils.X_validation)
      np.save(os.path.join(self.data_path, 'Data/X_test.npy'), self.utils.X_test)
      np.save(os.path.join(self.data_path, 'Data/y_train.npy'), self.utils.y_train)
      np.save(os.path.join(self.data_path, 'Data/y_validation.npy'), self.utils.y_validation)
      np.save(os.path.join(self.data_path, 'Data/y_test.npy'), self.utils.y_test)
      print("Data Processing Completed")
    if self.utils.continue_training == True:
        custom_objects = {'tversky_loss': self.tversky_loss, 'cbam_block': self.cbam_block, 'Channel_attention': Channel_attention, 'Position_attention': Position_attention}
        unet = load_model('saved_models/r2_atrous_unet_plus_plus_best_model.h5', custom_objects=custom_objects)
        unet.summary()
        if self.utils.override_trained_optimizer:
            learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
            if self.utils.deep_supervision == True:
                unet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer_1': self.tversky_loss, 'output_layer_2': self.tversky_loss, 'output_layer_3': self.tversky_loss, 'output_layer_4': self.tversky_loss}, metrics={'output_layer_1': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_2': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_3': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_4': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)})
            else:
                unet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer_4': self.tversky_loss}, metrics={'output_layer_4': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)})
    else:
        unet = self.build_3d_r2unet_plus_plus()
        unet.summary()
    print("Training Begins...")
    unet.fit(x = self.utils.X_train, y = self.utils.y_train, batch_size = self.utils.batch_size, epochs=self.utils.num_epochs, validation_data=(self.utils.X_validation, self.utils.y_validation), callbacks=[tf.keras.callbacks.ModelCheckpoint("saved_models/r2_atrous_unet_plus_plus_best_model.h5", save_best_only=True), tf.keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)])
    print("Training Ended, Model Saved!")
    return None

  def predict(self, new_data = None):
    '''
    This method will take a pre-trained model and make corresponding predictions.
    '''
    unet = self.build_3d_r2unet_plus_plus()
    unet.load_weights('saved_models/r2_atrous_unet_plus_plus_best_model.h5')
    if new_data is not None:
      n_features = self.utils.n_features
      if self.utils.svd_denoising == True:
          print("Raw Data Denoising...")
          denoised_data, variance_matrix = self.utils.svd_denoise(new_data, n_svd = self.utils.n_svd)
          # Compute cumulative variance explained
          cumulative_variance = np.cumsum(variance_matrix**2) / np.sum(variance_matrix**2)
          while True:
            if cumulative_variance[self.utils.n_svd - 1] <= self.utils.svd_denoise_threshold:
              self.utils.n_svd += 1
              denoised_data, variance_matrix = self.utils.svd_denoise(new_data, n_svd = self.utils.n_svd)
              cumulative_variance = np.cumsum(variance_matrix**2) / np.sum(variance_matrix**2)
            else:
              new_data = denoised_data
              break
          print("Raw Data Denoise Completed")
      else:
          pass
      if self.utils.layer_standardization == True:
        print("Layer-Wise Standardization...")
        X_normalized = self.utils.run_layer_standardization(new_data)
        print("Layer-Wise Standardization Completed")
      else:
        print("Layer-Wise Normalization...")
        X_normalized = self.utils.run_layer_normalization(new_data)
        print("Layer-Wise Normalization Completed")
      X_, pca = self.utils.run_PCA(image_cube = X_normalized, num_principal_components = self.utils.num_components_to_keep)
      X_ = cv2.resize(X_, (self.utils.resized_x_y, self.utils.resized_x_y), interpolation = cv2.INTER_LANCZOS4)
      X_test = X_.reshape(-1, self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep, 1)
      prediction_result = unet.predict(X_test)[3]
      prediction_encoded = np.zeros((self.utils.resized_x_y, self.utils.resized_x_y))
      for i in range(self.utils.resized_x_y):
        for j in range(self.utils.resized_x_y):
            prediction_encoded[i][j] = np.argmax(prediction_result[0][i][j])

      prediction = cv2.resize(prediction_encoded, (new_data.shape[1], new_data.shape[0]), interpolation = cv2.INTER_NEAREST)
      return prediction
    else:
      if self.utils.pre_load_dataset == True:
        print("Loading Testing Dataset...")
        self.utils.X_test = np.load(os.path.join(self.data_path, 'Data/X_test.npy')).astype(np.float64)
        self.utils.y_test = np.load(os.path.join(self.data_path, 'Data/y_test.npy')).astype(np.float64)
        print("Testing Dataset Loaded")
      else:
        print("Testing Begins...")
      total_test_length = (self.utils.X_test.shape[0]//self.utils.batch_size)*self.utils.batch_size
      prediction_result = np.zeros(shape=(total_test_length, self.utils.resized_x_y, self.utils.resized_x_y, self.utils.n_features))
      for i in range(0, total_test_length, self.utils.batch_size):
        print("Testing sample from:", i, "to:", i+self.utils.batch_size)
        prediction_result[i:i+self.utils.batch_size] = unet.predict(self.utils.X_test[i:i+self.utils.batch_size])[3]

      prediction_ = np.zeros(shape=(total_test_length, self.utils.resized_x_y, self.utils.resized_x_y))
      for k in range(total_test_length):
        prediction_encoded = np.zeros((self.utils.resized_x_y, self.utils.resized_x_y))
        for i in range(self.utils.resized_x_y):
          for j in range(self.utils.resized_x_y):
            prediction_encoded[i][j] = np.argmax(prediction_result[k][i][j])
        prediction = cv2.resize(prediction_encoded, (self.utils.X_test[k].shape[1], self.utils.X_test[k].shape[0]), interpolation = cv2.INTER_LANCZOS4)
        prediction_[k] = prediction
      print("Testing Ends...")
      return prediction_, np.argmax(self.utils.y_test[0:total_test_length], axis=-1)