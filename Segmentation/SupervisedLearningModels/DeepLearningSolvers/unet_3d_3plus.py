from utils import utils
import tensorflow as tf
from tensorflow import reduce_sum
import tensorflow_probability as tfp
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Reshape, concatenate, Conv2D, MaxPooling3D, UpSampling3D, BatchNormalization, Add, Activation, Dropout, UpSampling2D, Flatten, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

class UNet3Plus:
  """
  Class for Deep Learning Hyperspectral Segmentation with the 2020 UNet 3+ by Huang et.al.
  Input:  utils: the utilility class
  """
  def __init__(self, utils):
    self.utils = utils
  
  def encoding_layers_building_blocks(self, units, in_layer, pooling_layer=True):
      '''
      This method returns the encoding layers building blocks (i.e, downsampling layers)
      '''
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(in_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('elu')(convolution_layer)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('elu')(convolution_layer)
      if pooling_layer == True:
          out_layer = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(convolution_layer)
          return out_layer, convolution_layer
      else:
          return convolution_layer
    
  def decoding_layers_building_blocks(self, units, in_layer, is_bn = True, is_activation = True):
      '''
      This method returns the decoding layers building blocks (i.e, upsampling layers)
      '''
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(in_layer)
      if is_bn:
          convolution_layer = BatchNormalization()(convolution_layer)
      if is_activation:
          convolution_layer = Activation('elu')(convolution_layer)
      return convolution_layer
      
  def iou(self, y_true, y_pred, smooth=1):
    """
    Calculate intersection over union (IoU) between images.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])
    union = union - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

  def iou_loss(self, y_true, y_pred):
    """
    Jaccard / IoU loss
    """
    return 1 - self.iou(y_true, y_pred)

  def focal_loss(self, y_true, y_pred):
    """
    Focal loss
    """
    gamma = 2.
    alpha = 4.
    epsilon = 1.e-9

    y_true_c = tf.convert_to_tensor(y_true, tf.float32)
    y_pred_c = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred_c, epsilon)
    ce = tf.multiply(y_true_c, -tf.math.log(model_out))
    weight = tf.multiply(y_true_c, tf.pow(
        tf.subtract(1., model_out), gamma)
                         )
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=-1)
    return tf.reduce_mean(reduced_fl)

  def ssim_loss(self, y_true, y_pred):
    """
    Structural Similarity Index loss.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1)
    return K.mean(1 - ssim_value, axis=0)

  def dice_coef(self, y_true, y_pred, smooth=1.e-9):
    """
    Calculate dice coefficient.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
  
  def unet3p_hybrid_loss(self, y_true, y_pred):
    """
    Hybrid loss proposed in
    UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
    Hybrid loss for segmentation in three-level hierarchy – pixel,
    patch and map-level, which is able to capture both large-scale
    and fine structures with clear boundaries.
    """
    f_loss = self.focal_loss(y_true, y_pred)
    ms_ssim_loss = self.ssim_loss(y_true, y_pred)
    jacard_loss = self.iou_loss(y_true, y_pred)

    return f_loss + ms_ssim_loss + jacard_loss
    
  def build_3d_unet_3plus(self):
    '''
    This function is a tensorflow realization of 3D-U-Net 3+ Model (2020).
    '''
    input_layer = Input((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep, 1))
    # down sampling blocks
    down_sampling_pooling_layer_1, down_sampling_convolution_layer_1 = self.encoding_layers_building_blocks(64, input_layer)
    down_sampling_pooling_layer_2, down_sampling_convolution_layer_2 = self.encoding_layers_building_blocks(128, down_sampling_pooling_layer_1)
    down_sampling_pooling_layer_3, down_sampling_convolution_layer_3 = self.encoding_layers_building_blocks(256, down_sampling_pooling_layer_2)
    down_sampling_pooling_layer_4, down_sampling_convolution_layer_4 = self.encoding_layers_building_blocks(512, down_sampling_pooling_layer_3)
    # encoding blocks
    encoding_space_output_layer = self.encoding_layers_building_blocks(1024, down_sampling_pooling_layer_4, pooling_layer=False)
    
    cat_channels = 64
    cat_blocks = 5
    upsample_channels = cat_blocks * cat_channels
    
    # up sampling blocks
    # d4
    e1_d4 = MaxPooling3D(pool_size=(8, 8, 8), padding='same')(down_sampling_convolution_layer_1)
    e1_d4 = self.decoding_layers_building_blocks(cat_channels, e1_d4)
    
    e2_d4 = MaxPooling3D(pool_size=(4, 4, 4), padding='same')(down_sampling_convolution_layer_2)
    e2_d4 = self.decoding_layers_building_blocks(cat_channels, e2_d4)

    e3_d4 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(down_sampling_convolution_layer_3)
    e3_d4 = self.decoding_layers_building_blocks(cat_channels, e3_d4)

    e4_d4 = self.decoding_layers_building_blocks(cat_channels, down_sampling_convolution_layer_4)

    e5_d4 = UpSampling3D(size=(2, 2, 2))(encoding_space_output_layer)
    e5_d4 = self.decoding_layers_building_blocks(cat_channels, e5_d4)

    d4 = concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4], axis=3)
    d4 = self.decoding_layers_building_blocks(upsample_channels, d4)

    # d3
    e1_d3 = MaxPooling3D(pool_size=(4, 4, 4), padding='same')(down_sampling_convolution_layer_1)
    e1_d3 = self.decoding_layers_building_blocks(cat_channels, e1_d3)
    
    e2_d3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(down_sampling_convolution_layer_2)
    e2_d3 = self.decoding_layers_building_blocks(cat_channels, e2_d3)

    e3_d3 = self.decoding_layers_building_blocks(cat_channels, down_sampling_convolution_layer_3)

    e4_d3 = UpSampling3D(size=(2, 2, 2))(d4)
    e4_d3 = self.decoding_layers_building_blocks(cat_channels, e4_d3)

    e5_d3 = UpSampling3D(size=(4, 4, 4))(encoding_space_output_layer)
    e5_d3 = self.decoding_layers_building_blocks(cat_channels, e5_d3)

    d3 = concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3], axis=3)
    d3 = self.decoding_layers_building_blocks(upsample_channels, d3)

    # d2
    e1_d2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(down_sampling_convolution_layer_1)
    e1_d2 = self.decoding_layers_building_blocks(cat_channels, e1_d2)
    
    e2_d2 = self.decoding_layers_building_blocks(cat_channels, down_sampling_convolution_layer_2)

    e3_d2 = UpSampling3D(size=(2, 2, 2))(d3)
    e3_d2 = self.decoding_layers_building_blocks(cat_channels, e3_d2)

    e4_d2 = UpSampling3D(size=(4, 4, 4))(d4)
    e4_d2 = self.decoding_layers_building_blocks(cat_channels, e4_d2)

    e5_d2 = UpSampling3D(size=(8, 8, 8))(encoding_space_output_layer)
    e5_d2 = self.decoding_layers_building_blocks(cat_channels, e5_d2)

    d2 = concatenate([e1_d2, e2_d2, e3_d2, e4_d2, e5_d2], axis=3)
    d2 = self.decoding_layers_building_blocks(upsample_channels, d2)
    
    # d1
    e1_d1 = self.decoding_layers_building_blocks(cat_channels, down_sampling_convolution_layer_1)
    
    e2_d1 = UpSampling3D(size=(2, 2, 2))(d2)
    e2_d1 = self.decoding_layers_building_blocks(cat_channels, e2_d1)

    e3_d1 = UpSampling3D(size=(4, 4, 4))(d3)
    e3_d1 = self.decoding_layers_building_blocks(cat_channels, e3_d1)

    e4_d1 = UpSampling3D(size=(8, 8, 8))(d4)
    e4_d1 = self.decoding_layers_building_blocks(cat_channels, e4_d1)

    e5_d1 = UpSampling3D(size=(16, 16, 16))(encoding_space_output_layer)
    e5_d1 = self.decoding_layers_building_blocks(cat_channels, e5_d1)

    d1 = concatenate([e1_d1, e2_d1, e3_d1, e4_d1, e5_d1], axis=3)
    d1 = self.decoding_layers_building_blocks(upsample_channels, d1)
    
    # classification blocks
    up_sampling_output_layer_1_shape = d1.shape
    up_sampling_output_layer_1_2d_reshaped = Reshape((up_sampling_output_layer_1_shape[1], up_sampling_output_layer_1_shape[2], up_sampling_output_layer_1_shape[3] * up_sampling_output_layer_1_shape[4]))(d1)
    d1 = Conv2D(self.utils.n_features, (1, 1))(up_sampling_output_layer_1_2d_reshaped)
    d1 = Activation('softmax', name='output_layer_1')(d1)

    """Deep Supervision Part"""
    if self.utils.deep_supervision:
        up_sampling_output_layer_2_shape = d2.shape
        up_sampling_output_layer_2_2d_reshaped = Reshape((up_sampling_output_layer_2_shape[1], up_sampling_output_layer_2_shape[2], up_sampling_output_layer_2_shape[3] * up_sampling_output_layer_2_shape[4]))(d2)
        output_layer_2 = Conv2D(self.utils.n_features, (1, 1))(up_sampling_output_layer_2_2d_reshaped)
        d2 =UpSampling2D(size=(2, 2), interpolation='bilinear')(output_layer_2)
        d2 = Activation('softmax', name='output_layer_2')(d2)
    
        up_sampling_output_layer_3_shape = d3.shape
        up_sampling_output_layer_3_2d_reshaped = Reshape((up_sampling_output_layer_3_shape[1], up_sampling_output_layer_3_shape[2], up_sampling_output_layer_3_shape[3] * up_sampling_output_layer_3_shape[4]))(d3)
        output_layer_3 = Conv2D(self.utils.n_features, (1, 1))(up_sampling_output_layer_3_2d_reshaped)
        d3 =UpSampling2D(size=(4, 4), interpolation='bilinear')(output_layer_3)
        d3 = Activation('softmax', name='output_layer_3')(d3)
        
        up_sampling_output_layer_4_shape = d4.shape
        up_sampling_output_layer_4_2d_reshaped = Reshape((up_sampling_output_layer_4_shape[1], up_sampling_output_layer_4_shape[2], up_sampling_output_layer_4_shape[3] * up_sampling_output_layer_4_shape[4]))(d4)
        output_layer_4 = Conv2D(self.utils.n_features, (1, 1))(up_sampling_output_layer_4_2d_reshaped)
        d4 =UpSampling2D(size=(8, 8), interpolation='bilinear')(output_layer_4)
        d4 = Activation('softmax', name='output_layer_4')(d4)
        
        up_sampling_output_layer_5_shape = encoding_space_output_layer.shape
        up_sampling_output_layer_5_2d_reshaped = Reshape((up_sampling_output_layer_5_shape[1], up_sampling_output_layer_5_shape[2], up_sampling_output_layer_5_shape[3] * up_sampling_output_layer_5_shape[4]))(encoding_space_output_layer)
        output_layer_5 = Conv2D(self.utils.n_features, (1, 1))(up_sampling_output_layer_5_2d_reshaped)
        encoding_space_output_layer =UpSampling2D(size=(16, 16), interpolation='bilinear')(output_layer_5)
        e5 = Activation('softmax', name='output_layer_5')(encoding_space_output_layer)
        
        model = Model(inputs=[input_layer], outputs=[d1, d2, d3, d4, e5])
        learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer_1': self.unet3p_hybrid_loss, 'output_layer_2': self.unet3p_hybrid_loss, 'output_layer_3': self.unet3p_hybrid_loss, 'output_layer_4': self.unet3p_hybrid_loss, 'output_layer_5': self.unet3p_hybrid_loss}, metrics={'output_layer_1': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_2': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_3': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_4': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features), 'output_layer_5': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)})
    else:
        model = Model(inputs=[input_layer], outputs=[d1])
        learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer_1': self.unet3p_hybrid_loss}, metrics={'output_layer_1': tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)})
    return model

  def train(self):
    '''
    This method internally train the U-Net defined above, with a given set of hyperparameters.
    '''
    if self.utils.pre_load_dataset == True:
      print("Loading Training & Validation Dataset...")
      self.utils.X_train = np.load('X_train.npy').astype(np.float64)
      self.utils.X_validation = np.load('X_validation.npy').astype(np.float64)
      self.utils.y_train = np.load('y_train.npy').astype(np.float64)
      self.utils.y_validation = np.load('y_validation.npy').astype(np.float64)
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
      X_validation, X_test, y_validation, y_test = train_test_split(X_, y_, train_size = 1 - self.utils.test_ratio, test_size = self.utils.test_ratio, random_state=1234)
      self.utils.X_train, self.utils.X_validation, self.utils.X_test = X_train, X_validation, X_test
      self.utils.y_train, self.utils.y_validation, self.utils.y_test = y_train, y_validation, y_test
      np.save('X_train.npy', self.utils.X_train)
      np.save('X_validation.npy', self.utils.X_validation)
      np.save('X_test.npy', self.utils.X_test)
      np.save('y_train.npy', self.utils.y_train)
      np.save('y_validation.npy', self.utils.y_validation)
      np.save('y_test.npy', self.utils.y_test)
      print("Data Processing Completed")
    if self.utils.continue_training == True:
        # Custom objects, if any (you might need to define them depending on the custom loss, metrics, etc.)
        custom_objects = {'MeanIoU': tf.keras.metrics.MeanIoU}
        # Load the full model, including optimizer state
        unet = load_model('models/unet_3plus_best_model.h5', custom_objects=custom_objects)
        unet.summary()
    else:
        unet = self.build_3d_unet_3plus()
        unet.summary()
    print("Training Begins...")
    unet.fit(x = self.utils.X_train, y = self.utils.y_train, batch_size = self.utils.batch_size, epochs=self.utils.num_epochs, validation_data=(self.utils.X_validation, self.utils.y_validation), callbacks=[tf.keras.callbacks.ModelCheckpoint("models/unet_3plus_best_model.h5", save_best_only=True), tf.keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)])
    print("Training Ended, Model Saved!")
    return None

  def predict(self, new_data = None):
    '''
    This method will take a pre-trained model and make corresponding predictions.
    '''
    unet = self.build_3d_unet_3plus()
    unet.load_weights('models/unet_3plus_best_model.h5')
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
      prediction_result = unet.predict(X_test)
      prediction_encoded = np.zeros((self.utils.resized_x_y, self.utils.resized_x_y))
      for i in range(self.utils.resized_x_y):
        for j in range(self.utils.resized_x_y):
            prediction_encoded[i][j] = np.argmax(prediction_result[0][i][j])

      prediction = cv2.resize(prediction_encoded, (new_data.shape[1], new_data.shape[0]), interpolation = cv2.INTER_NEAREST)
      return prediction
    else:
      if self.utils.pre_load_dataset == True:
        print("Loading Testing Dataset...")
        self.utils.X_test = np.load('X_test.npy').astype(np.float64)
        self.utils.y_test = np.load('y_test.npy').astype(np.float64)
        print("Testing Dataset Loaded")
      else:
        print("Testing Begins...")
      total_test_length = (self.utils.X_test.shape[0]//self.utils.batch_size)*self.utils.batch_size
      prediction_result = np.zeros(shape=(total_test_length, self.utils.resized_x_y, self.utils.resized_x_y, self.utils.n_features))
      for i in range(0, total_test_length, self.utils.batch_size):
        print("Testing sample from:", i, "to:", i+self.utils.batch_size)
        prediction_result[i:i+self.utils.batch_size] = unet.predict(self.utils.X_test[i:i+self.utils.batch_size])

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