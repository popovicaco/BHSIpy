from utils import utils
import tensorflow as tf
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Reshape, concatenate, Conv2D, MaxPooling3D, UpSampling3D, BatchNormalization, Add, Activation, Dropout, Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

class MaxPoolingWithArgmax2D(Layer):
    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, *pool_size, 1]
        padding = padding.upper()
        strides = [1, *strides, 1]
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=ksize,
            strides=strides,
            padding=padding)

        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        mask = K.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')

        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.size[0],
                input_shape[2] * self.size[1],
                input_shape[3])

        ret = tf.scatter_nd(K.expand_dims(K.flatten(mask)),
                            K.flatten(updates),
                            [K.prod(output_shape)])

        input_shape = updates.shape
        out_shape = [-1,
                     input_shape[1] * self.size[0],
                     input_shape[2] * self.size[1],
                     input_shape[3]]
        return K.reshape(ret, out_shape)

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3]
        )

class SegNet:
  """
  Class for Deep Learning Hyperspectral Segmentation with a SegNet by Badrinarayanan et.al. (2016), the architecture is similar to a VGG16.
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
      convolution_layer = Activation('relu')(convolution_layer)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      if pooling_layer == True:
          pooling_layer = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(convolution_layer)
          convolution_layer_2d_reshaped = Reshape((convolution_layer.shape[1], convolution_layer.shape[2], convolution_layer.shape[3] * convolution_layer.shape[4]))(convolution_layer)
          _, mask = MaxPoolingWithArgmax2D(pool_size=(2, 2), strides=(2, 2))(convolution_layer_2d_reshaped)
          return pooling_layer, convolution_layer, mask
      else:
          return convolution_layer
    
  def encoding_layers_building_blocks_enhanced(self, units, in_layer, pooling_layer=True):
      '''
      This method returns the encoding layers building blocks (i.e, downsampling layers)
      '''
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(in_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      if pooling_layer == True:
          pooling_layer = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(convolution_layer)
          convolution_layer_2d_reshaped = Reshape((convolution_layer.shape[1], convolution_layer.shape[2], convolution_layer.shape[3] * convolution_layer.shape[4]))(convolution_layer)
          _, mask = MaxPoolingWithArgmax2D(pool_size=(2, 2), strides=(2, 2))(convolution_layer_2d_reshaped)
          return pooling_layer, convolution_layer, mask
      else:
          return convolution_layer
    
  def decoding_layers_building_blocks(self, units, in_layer, concat_layer):
      '''
      This method returns the decoding layers building blocks (i.e, upsampling layers)
      '''
      shape_3d = in_layer.shape
      in_layer_2d_reshaped = Reshape((shape_3d[1], shape_3d[2], shape_3d[3] * shape_3d[4]))(in_layer)
      upsampling_layer_2d = MaxUnpooling2D(size=(2, 2))([in_layer_2d_reshaped, concat_layer])
      upsampling_layer = Reshape((2*shape_3d[1], 2*shape_3d[2], shape_3d[3], shape_3d[4]))(upsampling_layer_2d)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(upsampling_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      out_layer = Activation('relu')(convolution_layer)
      return out_layer
      
  def decoding_layers_building_blocks_enhanced(self, units, in_layer, concat_layer):
      '''
      This method returns the decoding layers building blocks (i.e, upsampling layers)
      '''
      shape_3d = in_layer.shape
      in_layer_2d_reshaped = Reshape((shape_3d[1], shape_3d[2], shape_3d[3] * shape_3d[4]))(in_layer)
      upsampling_layer_2d = MaxUnpooling2D(size=(2, 2))([in_layer_2d_reshaped, concat_layer])
      upsampling_layer = Reshape((2*shape_3d[1], 2*shape_3d[2], shape_3d[3], shape_3d[4]))(upsampling_layer_2d)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(upsampling_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      convolution_layer = Conv3D(units, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      convolution_layer = Activation('relu')(convolution_layer)
      convolution_layer = Conv3D(units // 2, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer)
      convolution_layer = BatchNormalization()(convolution_layer)
      out_layer = Activation('relu')(convolution_layer)
      return out_layer
      
  def build_3d_segnet(self):
    '''
    This function is a tensorflow realization of the SegNet (2016). This is the version with Encoder Addition, to get the best result.
    '''
    input_layer = Input((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep, 1))
    # down sampling blocks
    down_sampling_output_layer_1, down_sampling_convolution_layer_1, down_sampling_indicies_layer_1 = self.encoding_layers_building_blocks(64, input_layer)
    down_sampling_output_layer_2, down_sampling_convolution_layer_2, down_sampling_indicies_layer_2 = self.encoding_layers_building_blocks(128, down_sampling_output_layer_1)
    down_sampling_output_layer_3, down_sampling_convolution_layer_3, down_sampling_indicies_layer_3 = self.encoding_layers_building_blocks_enhanced(256, down_sampling_output_layer_2)
    down_sampling_output_layer_4, down_sampling_convolution_layer_4, down_sampling_indicies_layer_4 = self.encoding_layers_building_blocks_enhanced(512, down_sampling_output_layer_3)
    # encoding blocks
    down_sampling_output_layer_5, down_sampling_convolution_layer_5, down_sampling_indicies_layer_5 = self.encoding_layers_building_blocks_enhanced(1024, down_sampling_output_layer_4)
    
    up_sampling_output_layer_1 = self.decoding_layers_building_blocks_enhanced(1024, down_sampling_output_layer_5, down_sampling_indicies_layer_5)
    # up sampling blocks
    up_sampling_output_layer_2 = self.decoding_layers_building_blocks_enhanced(512, up_sampling_output_layer_1, down_sampling_indicies_layer_4)
    up_sampling_output_layer_3 = self.decoding_layers_building_blocks(256, up_sampling_output_layer_2, down_sampling_indicies_layer_3)
    up_sampling_output_layer_3 = Conv3D(256, (3, 3, 3), padding='same', kernel_initializer='he_normal')(up_sampling_output_layer_3)
    up_sampling_output_layer_3 = BatchNormalization()(up_sampling_output_layer_3)
    up_sampling_output_layer_3 = Activation('relu')(up_sampling_output_layer_3)
    
    up_sampling_output_layer_4 = self.decoding_layers_building_blocks(128, up_sampling_output_layer_3, down_sampling_indicies_layer_2)
    up_sampling_output_layer_4 = Conv3D(256, (3, 3, 3), padding='same', kernel_initializer='he_normal')(up_sampling_output_layer_4)
    up_sampling_output_layer_4 = BatchNormalization()(up_sampling_output_layer_4)
    up_sampling_output_layer_4 = Activation('relu')(up_sampling_output_layer_4)
    
    up_sampling_output_layer_5 = self.decoding_layers_building_blocks(64, up_sampling_output_layer_4, down_sampling_indicies_layer_1)
    up_sampling_output_layer_5 = Conv3D(256, (3, 3, 3), padding='same', kernel_initializer='he_normal')(up_sampling_output_layer_5)
    up_sampling_output_layer_5 = BatchNormalization()(up_sampling_output_layer_5)
    up_sampling_output_layer_5 = Activation('relu')(up_sampling_output_layer_5)
    
    # classification block
    up_sampling_output_layer_5_shape = up_sampling_output_layer_5.shape
    up_sampling_output_layer_5_2d_reshaped = Reshape((up_sampling_output_layer_5_shape[1], up_sampling_output_layer_5_shape[2], up_sampling_output_layer_5_shape[3] * up_sampling_output_layer_5_shape[4]))(up_sampling_output_layer_5)
    output_layer = Conv2D(self.utils.n_features, (1, 1), activation='softmax', name='output_layer')(up_sampling_output_layer_5_2d_reshaped)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer': 'categorical_crossentropy'}, metrics=[tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)])
    return model

  def train(self):
    '''
    This method internally train the Seg-Net defined above, with a given set of hyperparameters.
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
        custom_objects = {'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D, 'MaxUnpooling2D': MaxUnpooling2D}
        # Load the full model, including optimizer state
        segnet = load_model('models/segnet_best_model.h5', custom_objects=custom_objects)
        segnet.summary()
        if self.utils.override_trained_optimizer:
            learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
            segnet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer': 'categorical_crossentropy'}, metrics=[tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)])
    else:
        segnet = self.build_3d_segnet()
        segnet.summary()
    print("Training Begins...")
    segnet.fit(x = self.utils.X_train, y = self.utils.y_train, batch_size = self.utils.batch_size, epochs=self.utils.num_epochs, validation_data=(self.utils.X_validation, self.utils.y_validation), callbacks=[tf.keras.callbacks.ModelCheckpoint("models/segnet_best_model.h5", save_best_only=True), tf.keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)])
    print("Training Ended, Model Saved!")
    return None

  def predict(self, new_data = None):
    '''
    This method will take a pre-trained model and make corresponding predictions.
    '''
    segnet = self.build_3d_segnet()
    segnet.load_weights('models/segnet_best_model.h5')
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
      prediction_result = segnet.predict(X_test)
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
        prediction_result[i:i+self.utils.batch_size] = segnet.predict(self.utils.X_test[i:i+self.utils.batch_size])

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