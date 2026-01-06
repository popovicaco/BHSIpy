from re import I
import os
from utils import utils
import tensorflow as tf
from tensorflow import reduce_sum
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Reshape, concatenate, Conv2D, MaxPooling3D, Conv3DTranspose, BatchNormalization, Add, Activation, Dropout, PReLU, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

class VNet:
  """
  Class for Deep Learning Hyperspectral Segmentation with the V-Net by Milletari (2016).
  Input:  utils: the utilility class
  """
  def __init__(self, utils):
    self.utils = utils
    self.data_path = os.path.join(os.getcwd(), '')
  
  def encoding_layers_building_blocks(self, units, in_layer, downsampling_layer=True):
      '''
      This method returns the encoding layers building blocks (i.e, downsampling layers)
      '''
      skip_connection = in_layer
      convolution_layer = Conv3D(units, (5, 5, 5), padding='same', kernel_initializer='he_normal')(in_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = PReLU()(convolution_layer)
      convolution_layer = Conv3D(units, (5, 5, 5), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = PReLU()(convolution_layer)
      convolution_layer = Conv3D(units, (5, 5, 5), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = PReLU()(convolution_layer)
      convolution_layer = Add()([convolution_layer, skip_connection])
      if downsampling_layer == True:
          out_layer = Conv3D(units*2, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(convolution_layer)
          return out_layer, convolution_layer
      else:
          return convolution_layer
    
  def decoding_layers_building_blocks(self, units, in_layer, concat_layer):
      '''
      This method returns the decoding layers building blocks (i.e, upsampling layers)
      '''
      if concat_layer.shape[3] == 1:
          upsampling_layer = Conv3DTranspose(units, (2, 2, 1), strides=(2, 2, 1), padding='same', kernel_initializer='he_normal')(in_layer)
          skip_connection = upsampling_layer
      else:
          upsampling_layer = Conv3DTranspose(units, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(in_layer)
          skip_connection = upsampling_layer
      upsampling_layer = concatenate([upsampling_layer, concat_layer], axis=4)
      convolution_layer = Conv3D(units, (5, 5, 5), padding='same', kernel_initializer='he_normal')(upsampling_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = PReLU()(convolution_layer)
      convolution_layer = Conv3D(units, (5, 5, 5), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      out_layer = PReLU()(convolution_layer)
      convolution_layer = Conv3D(units, (5, 5, 5), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      out_layer = PReLU()(convolution_layer)
      convolution_layer = Add()([convolution_layer, skip_connection])
      return out_layer
    
  def dice_loss(self, y_true, y_pred, smooth=1e-6):
    # Calculate intersection and the sum for the numerator and denominator of the Dice score
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_true_pred = K.sum(y_true, axis=[0, 1, 2]) + K.sum(y_pred, axis=[0, 1, 2])

    # Calculate the Dice score for each class
    dice_scores = (2. * intersection + smooth) / (sum_true_pred + smooth)
    dice_loss_multiclass = 1 - K.mean(dice_scores)
    return dice_loss_multiclass

  def build_3d_vnet(self):
    '''
    This function is a tensorflow realization of a 3D V-Net model (2016). 
    '''
    input_layer = Input((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep, 1))
    # down sampling blocks
    # 16 Channels
    skip_connection_1 = input_layer
    down_sampling_convolution_layer_1 = Conv3D(16, (5, 5, 5), padding='same', kernel_initializer='he_normal')(input_layer)
    down_sampling_convolution_layer_1 = BatchNormalization()(down_sampling_convolution_layer_1)
    down_sampling_convolution_layer_1 = PReLU()(down_sampling_convolution_layer_1)
    down_sampling_convolution_layer_1 = Add()([down_sampling_convolution_layer_1, skip_connection_1])
    down_sampling_output_layer_1 = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(down_sampling_convolution_layer_1)
    
    # 32 Channels
    skip_connection_2 = down_sampling_output_layer_1
    down_sampling_convolution_layer_2 = Conv3D(32, (5, 5, 5), padding='same', kernel_initializer='he_normal')(down_sampling_output_layer_1)
    down_sampling_convolution_layer_2 = BatchNormalization()(down_sampling_convolution_layer_2)
    down_sampling_convolution_layer_2 = PReLU()(down_sampling_convolution_layer_2)
    down_sampling_convolution_layer_2 = Conv3D(32, (5, 5, 5), padding='same', kernel_initializer='he_normal')(down_sampling_convolution_layer_2)
    down_sampling_convolution_layer_2 = BatchNormalization()(down_sampling_convolution_layer_2)
    down_sampling_convolution_layer_2 = PReLU()(down_sampling_convolution_layer_2)
    down_sampling_convolution_layer_2 = Add()([down_sampling_convolution_layer_2, skip_connection_2])
    down_sampling_output_layer_2 = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(down_sampling_convolution_layer_2)
    
    # 64 Channels
    down_sampling_output_layer_3, down_sampling_convolution_layer_3 = self.encoding_layers_building_blocks(64, down_sampling_output_layer_2)
    # 128 Channels
    down_sampling_output_layer_4, down_sampling_convolution_layer_4 = self.encoding_layers_building_blocks(128, down_sampling_output_layer_3)
    # encoding blocks, 256 Channels
    encoding_space_output_layer = self.encoding_layers_building_blocks(256, down_sampling_output_layer_4, downsampling_layer=False)
    
    # up sampling blocks
    # 128 Channels
    up_sampling_output_layer_1 = self.decoding_layers_building_blocks(128, encoding_space_output_layer, down_sampling_convolution_layer_4)
    # 64 Channels
    up_sampling_output_layer_2 = self.decoding_layers_building_blocks(64, up_sampling_output_layer_1, down_sampling_convolution_layer_3)
    # 32 Channels
    if down_sampling_convolution_layer_2.shape[3] == 1:
        upsampling_layer_3 = Conv3DTranspose(32, (2, 2, 1), strides=(2, 2, 1), padding='same', kernel_initializer='he_normal')(up_sampling_output_layer_2)
        skip_connection_up_3 = upsampling_layer_3
    else:
        upsampling_layer_3 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(up_sampling_output_layer_2)
        skip_connection_up_3 = upsampling_layer_3
    upsampling_layer_3 = concatenate([upsampling_layer_3, down_sampling_convolution_layer_2], axis=4)
    up_sampling_convolution_layer_3 = Conv3D(32, (5, 5, 5), padding='same', kernel_initializer='he_normal')(upsampling_layer_3)
    up_sampling_convolution_layer_3 = BatchNormalization()(up_sampling_convolution_layer_3)
    up_sampling_convolution_layer_3 = PReLU()(up_sampling_convolution_layer_3)
    up_sampling_convolution_layer_3 = Conv3D(32, (5, 5, 5), padding='same', kernel_initializer='he_normal')(up_sampling_convolution_layer_3)
    up_sampling_convolution_layer_3 = BatchNormalization()(up_sampling_convolution_layer_3)
    up_sampling_convolution_layer_3 = PReLU()(up_sampling_convolution_layer_3)
    up_sampling_output_layer_3 = Add()([up_sampling_convolution_layer_3, skip_connection_up_3])
    
    # 16 Channels
    if down_sampling_convolution_layer_1.shape[3] == 1:
        upsampling_layer_4 = Conv3DTranspose(16, (2, 2, 1), strides=(2, 2, 1), padding='same', kernel_initializer='he_normal')(up_sampling_output_layer_3)
        skip_connection_up_4 = upsampling_layer_4
    else:
        upsampling_layer_4 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(up_sampling_output_layer_3)
        skip_connection_up_4 = upsampling_layer_4
    upsampling_layer_4 = concatenate([upsampling_layer_4, down_sampling_convolution_layer_1], axis=4)
    up_sampling_convolution_layer_4 = Conv3D(16, (5, 5, 5), padding='same', kernel_initializer='he_normal')(upsampling_layer_4)
    up_sampling_convolution_layer_4 = BatchNormalization()(up_sampling_convolution_layer_4)
    up_sampling_convolution_layer_4 = PReLU()(up_sampling_convolution_layer_4)
    up_sampling_output_layer_4 = Add()([up_sampling_convolution_layer_4, skip_connection_up_4])
    
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
    This method internally train the V-Net defined above, with a given set of hyperparameters.
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
        custom_objects = {'dice_loss': self.dice_loss}
        # Load the full model, including optimizer state
        vnet = load_model('models/vnet_best_model.h5', custom_objects=custom_objects)
        vnet.summary()
        if self.utils.override_trained_optimizer:
            learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
            vnet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer': self.dice_loss}, metrics=[tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)])
    else:
        vnet = self.build_3d_vnet()
        vnet.summary()
    print("Training Begins...")
    vnet.fit(x = self.utils.X_train, y = self.utils.y_train, batch_size = self.utils.batch_size, epochs=self.utils.num_epochs, validation_data=(self.utils.X_validation, self.utils.y_validation), callbacks=[tf.keras.callbacks.ModelCheckpoint("models/vnet_best_model.h5", save_best_only=True), tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)])
    print("Training Ended, Model Saved!")
    return None

  def predict(self, new_data = None):
    '''
    This method will take a pre-trained model and make corresponding predictions.
    '''
    vnet = self.build_3d_vnet()
    vnet.load_weights('models/vnet_best_model.h5')
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
      prediction_result = vnet.predict(X_test)
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
        prediction_result[i:i+self.utils.batch_size] = vnet.predict(self.utils.X_test[i:i+self.utils.batch_size])

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