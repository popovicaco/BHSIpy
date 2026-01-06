from re import I
import os
from utils import utils
import tensorflow as tf
from tensorflow import reduce_sum
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Reshape, concatenate, Conv2D, MaxPooling3D, Conv3DTranspose, BatchNormalization, Add, Activation, Dropout, Layer, InputSpec, AveragePooling3D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

class SwitchNormalization(Layer):
    """Switchable Normalization layer
    
    Citation: @article{punn2020inception,
              title={Inception u-net architecture for semantic segmentation to identify nuclei in microscopy cell images},
              author={Punn, Narinder Singh and Agarwal, Sonali},
              journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
              volume={16},
              number={1},
              pages={1--15},
              year={2020},
              publisher={ACM New York, NY, USA}
            }
            
    Switch Normalization performs Instance Normalization, Layer Normalization and Batch
    Normalization using its parameters, and then weighs them using learned parameters to
    allow different levels of interaction of the 3 normalization schemes for each layer.

    Only supports the moving average variant from the paper, since the `batch average`
    scheme requires dynamic graph execution to compute the mean and variance of several
    batches at runtime.

    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        momentum: Momentum for the moving mean and the moving variance. The original
            implementation suggests a default momentum of `0.997`, however it is highly
            unstable and training can fail after a few epochs. To stabilise training, use
            lower values of momentum such as `0.99` or `0.98`.
        epsilon: Small float added to variance to avoid dividing by zero.
        final_gamma: Bool value to determine if this layer is the final
            normalization layer for the residual block.  Overrides the initialization
            of the scaling weights to be `zeros`. Only used for Residual Networks,
            to make the forward/backward signal initially propagated through an
            identity shortcut.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        mean_weights_initializer: Initializer for the mean weights.
        variance_weights_initializer: Initializer for the variance weights.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        mean_weights_regularizer: Optional regularizer for the mean weights.
        variance_weights_regularizer: Optional regularizer for the variance weights.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
        mean_weights_constraints: Optional constraint for the mean weights.
        variance_weights_constraints: Optional constraint for the variance weights.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Differentiable Learning-to-Normalize via Switchable Normalization](https://arxiv.org/abs/1806.10779)
    """

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 final_gamma=False,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 mean_weights_initializer='ones',
                 variance_weights_initializer='ones',
                 moving_mean_initializer='ones',
                 moving_variance_initializer='zeros',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 mean_weights_regularizer=None,
                 variance_weights_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 mean_weights_constraints=None,
                 variance_weights_constraints=None,
                 **kwargs):
        super(SwitchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

        self.beta_initializer = initializers.get(beta_initializer)
        if final_gamma:
            self.gamma_initializer = initializers.get('zeros')
        else:
            self.gamma_initializer = initializers.get(gamma_initializer)
        self.mean_weights_initializer = initializers.get(mean_weights_initializer)
        self.variance_weights_initializer = initializers.get(variance_weights_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.mean_weights_regularizer = regularizers.get(mean_weights_regularizer)
        self.variance_weights_regularizer = regularizers.get(variance_weights_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.mean_weights_constraints = constraints.get(mean_weights_constraints)
        self.variance_weights_constraints = constraints.get(variance_weights_constraints)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name='gamma',
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name='beta',
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint)
        else:
            self.beta = None

        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)

        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)

        self.mean_weights = self.add_weight(
            shape=(3,),
            name='mean_weights',
            initializer=self.mean_weights_initializer,
            regularizer=self.mean_weights_regularizer,
            constraint=self.mean_weights_constraints)

        self.variance_weights = self.add_weight(
            shape=(3,),
            name='variance_weights',
            initializer=self.variance_weights_initializer,
            regularizer=self.variance_weights_regularizer,
            constraint=self.variance_weights_constraints)

        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]

        if self.axis != 0:
            del reduction_axes[0]

        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        mean_instance = K.mean(inputs, reduction_axes, keepdims=True)
        variance_instance = K.var(inputs, reduction_axes, keepdims=True)

        mean_layer = K.mean(mean_instance, self.axis, keepdims=True)
        temp = variance_instance + K.square(mean_instance)
        variance_layer = K.mean(temp, self.axis, keepdims=True) - K.square(mean_layer)

        def training_phase():
            mean_batch = K.mean(mean_instance, axis=0, keepdims=True)
            variance_batch = K.mean(temp, axis=0, keepdims=True) - K.square(mean_batch)

            mean_batch_reshaped = K.flatten(mean_batch)
            variance_batch_reshaped = K.flatten(variance_batch)

            if K.backend() != 'cntk':
                sample_size = K.prod([K.shape(inputs)[axis]
                                      for axis in reduction_axes])
                sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

                # sample variance - unbiased estimator of population variance
                variance_batch_reshaped *= sample_size / (sample_size - (1.0 + self.epsilon))

            self.add_update([K.moving_average_update(self.moving_mean, mean_batch_reshaped, self.momentum), K.moving_average_update(self.moving_variance, variance_batch_reshaped, self.momentum)])

            return normalize_func(mean_batch, variance_batch)

        def inference_phase():
            mean_batch = self.moving_mean
            variance_batch = self.moving_variance

            return normalize_func(mean_batch, variance_batch)

        def normalize_func(mean_batch, variance_batch):
            mean_batch = K.reshape(mean_batch, broadcast_shape)
            variance_batch = K.reshape(variance_batch, broadcast_shape)

            mean_weights = K.softmax(self.mean_weights, axis=0)
            variance_weights = K.softmax(self.variance_weights, axis=0)

            mean = (mean_weights[0] * mean_instance +
                    mean_weights[1] * mean_layer +
                    mean_weights[2] * mean_batch)

            variance = (variance_weights[0] * variance_instance +
                        variance_weights[1] * variance_layer +
                        variance_weights[2] * variance_batch)

            outputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

            if self.scale:
                broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                outputs = outputs * broadcast_gamma

            if self.center:
                broadcast_beta = K.reshape(self.beta, broadcast_shape)
                outputs = outputs + broadcast_beta

            return outputs

        if training in {0, False}:
            return inference_phase()

        return K.in_train_phase(training_phase,
                                inference_phase,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'mean_weights_initializer': initializers.serialize(self.mean_weights_initializer),
            'variance_weights_initializer': initializers.serialize(self.variance_weights_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'mean_weights_regularizer': regularizers.serialize(self.mean_weights_regularizer),
            'variance_weights_regularizer': regularizers.serialize(self.variance_weights_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'mean_weights_constraints': constraints.serialize(self.mean_weights_constraints),
            'variance_weights_constraints': constraints.serialize(self.variance_weights_constraints),
        }
        base_config = super(SwitchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

get_custom_objects().update({'SwitchNormalization': SwitchNormalization})

class Inception_UNet:
  """
  Class for Deep Learning Hyperspectral Segmentation with a 3D Inception UNet inspired from Punn (2020).
  Input:  utils: the utilility class
  """
  def __init__(self, utils):
    self.utils = utils
    self.data_path = os.path.join(os.getcwd(), '')
    
  def HybridPooling3D(self, units, pool_size=(2, 2, 2), padding='same'):
    def apply(layer):
        out_layer = concatenate([MaxPooling3D(pool_size=pool_size, padding=padding, strides=(1, 1, 1))(layer), AveragePooling3D(pool_size=pool_size, padding=padding, strides=(1, 1, 1))(layer)])
        return out_layer
    return apply
    
  def HybridPooling3D_DownSampling(self, pool_size=(2, 2, 2), padding='same'):
    def apply(layer):
        return concatenate([MaxPooling3D(pool_size=pool_size, padding=padding)(layer), AveragePooling3D(pool_size=pool_size, padding=padding)(layer)])
    return apply
  
  def encoding_layers_building_blocks(self, units, in_layer, pooling_layer=True):
      '''
      This method returns the encoding layers building blocks (i.e, downsampling layers)
      '''
      convolution_layer_1 = Conv3D(units, (1, 1, 1), padding='same', kernel_initializer='he_normal')(in_layer)
      convolution_layer_1 = BatchNormalization()(convolution_layer_1)
      convolution_layer_1 = Activation('relu')(convolution_layer_1)
      convolution_layer_2 = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(in_layer)
      convolution_layer_2 = BatchNormalization()(convolution_layer_2)
      convolution_layer_2 = Activation('relu')(convolution_layer_2)
      convolution_layer_3 = Conv3D(units, (5, 5, 5), padding='same', kernel_initializer='he_normal')(in_layer)
      convolution_layer_3 = BatchNormalization()(convolution_layer_3)
      convolution_layer_3 = Activation('relu')(convolution_layer_3)
      hybrid_pooling_layer = self.HybridPooling3D(units)(in_layer)
      concat_layer = concatenate([convolution_layer_1, convolution_layer_2, convolution_layer_3, hybrid_pooling_layer], axis=4)
      convolution_layer_4 = Conv3D(units, (1, 1, 1), padding='same', kernel_initializer='he_normal')(concat_layer)
      convolution_layer_4 = BatchNormalization()(convolution_layer_4)
      convolution_layer_4 = Activation('relu')(convolution_layer_4)
      drop_out_layer = Dropout(self.utils.dropout_rate)(convolution_layer_4)
      if pooling_layer == True:
          out_layer = self.HybridPooling3D_DownSampling(pool_size=(2, 2, 2), padding='same')(drop_out_layer)
          return out_layer, convolution_layer_4
      else:
          return convolution_layer_4
    
  def decoding_layers_building_blocks(self, units, in_layer, concat_layer):
      '''
      This method returns the decoding layers building blocks (i.e, upsampling layers)
      '''
      if concat_layer.shape[3] == 1:
          upsampling_layer = Conv3DTranspose(units, (2, 2, 1), strides=(2, 2, 1), padding='same', kernel_initializer='he_normal')(in_layer)
      else:
          upsampling_layer = Conv3DTranspose(units, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(in_layer)
      upsampling_layer = concatenate([upsampling_layer, concat_layer], axis=4)
      convolution_layer_1 = Conv3D(units, (1, 1, 1), padding='same', kernel_initializer='he_normal')(upsampling_layer)
      convolution_layer_1 = BatchNormalization()(convolution_layer_1)
      convolution_layer_1 = Activation('relu')(convolution_layer_1)
      convolution_layer_2 = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(upsampling_layer)
      convolution_layer_2 = BatchNormalization()(convolution_layer_2)
      convolution_layer_2 = Activation('relu')(convolution_layer_2)
      convolution_layer_3 = Conv3D(units, (5, 5, 5), padding='same', kernel_initializer='he_normal')(upsampling_layer)
      convolution_layer_3 = BatchNormalization()(convolution_layer_3)
      convolution_layer_3 = Activation('relu')(convolution_layer_3)
      hybrid_pooling_layer = self.HybridPooling3D(units)(upsampling_layer)
      concat_layer_ = concatenate([convolution_layer_1, convolution_layer_2, convolution_layer_3, hybrid_pooling_layer], axis=4)
      convolution_layer_4 = Conv3D(units, (1, 1, 1), padding='same', kernel_initializer='he_normal')(concat_layer_)
      convolution_layer_4 = BatchNormalization()(convolution_layer_4)
      convolution_layer_4 = Activation('relu')(convolution_layer_4)
      drop_out_layer = Dropout(self.utils.dropout_rate)(convolution_layer_4)
      return drop_out_layer
  
  def dice_loss(self, y_true, y_pred, smooth=1e-6):
    # Calculate intersection and the sum for the numerator and denominator of the Dice score
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_true_pred = K.sum(y_true, axis=[0, 1, 2]) + K.sum(y_pred, axis=[0, 1, 2])

    # Calculate the Dice score for each class
    dice_scores = (2. * intersection + smooth) / (sum_true_pred + smooth)
    dice_loss_multiclass = 1 - K.mean(dice_scores)
    return dice_loss_multiclass
      
  def build_3d_inception_unet(self):
    '''
    This function is a tensorflow realization of a newly designed 3D Inception U-Net model. 
    '''
    input_layer = Input((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep, 1))
    input_layer_ = SwitchNormalization(axis=-1)(input_layer)
    # down sampling blocks
    down_sampling_output_layer_1, down_sampling_convolution_layer_1 = self.encoding_layers_building_blocks(16, input_layer_)
    down_sampling_output_layer_2, down_sampling_convolution_layer_2 = self.encoding_layers_building_blocks(32, down_sampling_output_layer_1)
    down_sampling_output_layer_3, down_sampling_convolution_layer_3 = self.encoding_layers_building_blocks(64, down_sampling_output_layer_2)
    down_sampling_output_layer_4, down_sampling_convolution_layer_4 = self.encoding_layers_building_blocks(128, down_sampling_output_layer_3)
    # encoding blocks
    encoding_space_output_layer = self.encoding_layers_building_blocks(256, down_sampling_output_layer_4, pooling_layer=False)
    input_layer_down_sampled = self.HybridPooling3D_DownSampling(pool_size=(16, 16, 16), padding='same')(input_layer_)
    # up sampling blocks
    concat_layer = concatenate([encoding_space_output_layer, input_layer_down_sampled], axis=4)
    convolution_layer = Conv3D(256, (1, 1, 1), padding='same', kernel_initializer='he_normal')(concat_layer)
    up_sampling_output_layer_1 = self.decoding_layers_building_blocks(128, convolution_layer, down_sampling_convolution_layer_4)
    up_sampling_output_layer_2 = self.decoding_layers_building_blocks(64, up_sampling_output_layer_1, down_sampling_convolution_layer_3)
    up_sampling_output_layer_3 = self.decoding_layers_building_blocks(32, up_sampling_output_layer_2, down_sampling_convolution_layer_2)
    up_sampling_output_layer_4 = self.decoding_layers_building_blocks(16, up_sampling_output_layer_3, down_sampling_convolution_layer_1)
    # classification block
    output_layer = Conv3D(1, (1, 1, 1))(up_sampling_output_layer_4)
    output_layer = Reshape((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep))(output_layer)
    output_layer = Activation('softmax', name='output_layer')(output_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer': self.dice_loss}, metrics=[tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)])
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
        X_normalized = X
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
        # Custom objects, if any (you might need to define them depending on the custom loss, metrics, etc.)
        custom_objects = {'dice_loss': self.dice_loss, 'SwitchNormalization': SwitchNormalization}
        # Load the full model, including optimizer state
        inception_unet = load_model('saved_models/inception_unet_best_model.h5', custom_objects=custom_objects)
        inception_unet.summary()
        if self.utils.override_trained_optimizer:
            learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
            inception_unet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer': self.dice_loss}, metrics=[tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)])
    else:
        inception_unet = self.build_3d_inception_unet()
        inception_unet.summary()
    print("Training Begins...")
    inception_unet.fit(x = self.utils.X_train, y = self.utils.y_train, batch_size = self.utils.batch_size, epochs=self.utils.num_epochs, validation_data=(self.utils.X_validation, self.utils.y_validation), callbacks=[tf.keras.callbacks.ModelCheckpoint("saved_models/inception_unet_best_model.h5", save_best_only=True), tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)])
    print("Training Ended, Model Saved!")
    return None

  def predict(self, new_data = None):
    '''
    This method will take a pre-trained model and make corresponding predictions.
    '''
    inception_unet = self.build_3d_inception_unet()
    inception_unet.load_weights('saved_models/inception_unet_best_model.h5')
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
        X_normalized = new_data
      X_, pca = self.utils.run_PCA(image_cube = X_normalized, num_principal_components = self.utils.num_components_to_keep)
      X_ = cv2.resize(X_, (self.utils.resized_x_y, self.utils.resized_x_y), interpolation = cv2.INTER_LANCZOS4)
      X_test = X_.reshape(-1, self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep, 1)
      prediction_result = inception_unet.predict(X_test)
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
        prediction_result[i:i+self.utils.batch_size] = inception_unet.predict(self.utils.X_test[i:i+self.utils.batch_size])

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