from utils import utils
import tensorflow as tf
import tensorflow_probability as tfp
from skimage.transform import resize
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Reshape, concatenate, Conv2D, MaxPooling3D, Conv3DTranspose, BatchNormalization, Add, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

class UNet:
  """
  Class for Deep Learning Hyperspectral Segmentation with a simple 3D-UNet (the baseline model)
  Input:  utils: the utilility class
  """
  def __init__(self, utils):
    self.utils = utils

  def build_3d_unet(self):
    '''
    This function is a tensorflow realization of a newly designed Bayesian Residual 3D U-Net model. 
    '''
    input_layer = Input((self.utils.resized_x_y, self.utils.resized_x_y, self.utils.num_components_to_keep, 1))

    convolution_layer_1 = Conv3D(64, (3, 3, 3), padding='same', kernel_initializer='he_normal')(input_layer)
    convolution_layer_1 = BatchNormalization()(convolution_layer_1)
    convolution_layer_1 = Activation('relu')(convolution_layer_1)
    convolution_layer_1 = Conv3D(64, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_1)
    convolution_layer_1 = BatchNormalization()(convolution_layer_1)
    convolution_layer_1 = Activation('relu')(convolution_layer_1)
    pooling_layer_1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(convolution_layer_1)

    convolution_layer_2 = Conv3D(128, (3, 3, 3), padding='same', kernel_initializer='he_normal')(pooling_layer_1)
    convolution_layer_2 = BatchNormalization()(convolution_layer_2)
    convolution_layer_2 = Activation('relu')(convolution_layer_2)
    convolution_layer_2 = Conv3D(128, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_2)
    convolution_layer_2 = BatchNormalization()(convolution_layer_2)
    convolution_layer_2 = Activation('relu')(convolution_layer_2)
    pooling_layer_2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(convolution_layer_2)
    
    convolution_layer_3 = Conv3D(256, (3, 3, 3), padding='same', kernel_initializer='he_normal')(pooling_layer_2)
    convolution_layer_3 = BatchNormalization()(convolution_layer_3)
    convolution_layer_3 = Activation('relu')(convolution_layer_3)
    convolution_layer_3 = Conv3D(256, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_3)
    convolution_layer_3 = BatchNormalization()(convolution_layer_3)
    convolution_layer_3 = Activation('relu')(convolution_layer_3)
    pooling_layer_3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(convolution_layer_3)

    convolution_layer_4 = Conv3D(512, (3, 3, 3), padding='same', kernel_initializer='he_normal')(pooling_layer_3)
    convolution_layer_4 = BatchNormalization()(convolution_layer_4)
    convolution_layer_4 = Activation('relu')(convolution_layer_4)
    convolution_layer_4 = Conv3D(512, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_4)
    convolution_layer_4 = BatchNormalization()(convolution_layer_4)
    convolution_layer_4 = Activation('relu')(convolution_layer_4)
    pooling_layer_4 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(convolution_layer_4)
    
    convolution_layer_5 = Conv3D(1024, (3, 3, 3), padding='same', kernel_initializer='he_normal')(pooling_layer_4)
    convolution_layer_5 = BatchNormalization()(convolution_layer_5)
    convolution_layer_5 = Activation('relu')(convolution_layer_5)
    convolution_layer_5 = Conv3D(1024, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_5)
    convolution_layer_5 = BatchNormalization()(convolution_layer_5)
    convolution_layer_5 = Activation('relu')(convolution_layer_5)

    convolution_layer_6 = concatenate([Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(convolution_layer_5), convolution_layer_4], axis=3)
    convolution_layer_6 = Conv3D(512, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_6)
    convolution_layer_6 = BatchNormalization()(convolution_layer_6)
    convolution_layer_6 = Activation('relu')(convolution_layer_6)
    convolution_layer_6 = Conv3D(512, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_6)
    convolution_layer_6 = BatchNormalization()(convolution_layer_6)
    convolution_layer_6 = Activation('relu')(convolution_layer_6)

    convolution_layer_7 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(convolution_layer_6), convolution_layer_3], axis=3)
    convolution_layer_7 = Conv3D(256, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_7)
    convolution_layer_7 = BatchNormalization()(convolution_layer_7)
    convolution_layer_7 = Activation('relu')(convolution_layer_7)
    convolution_layer_7 = Conv3D(256, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_7)
    convolution_layer_7 = BatchNormalization()(convolution_layer_7)
    convolution_layer_7 = Activation('relu')(convolution_layer_7)

    convolution_layer_8 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(convolution_layer_7), convolution_layer_2], axis=3)
    convolution_layer_8 = Conv3D(128, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_8)
    convolution_layer_8 = BatchNormalization()(convolution_layer_8)
    convolution_layer_8 = Activation('relu')(convolution_layer_8)
    convolution_layer_8 = Conv3D(128, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_8)
    convolution_layer_8 = BatchNormalization()(convolution_layer_8)
    convolution_layer_8 = Activation('relu')(convolution_layer_8)
    
    convolution_layer_9 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', kernel_initializer='he_normal')(convolution_layer_8), convolution_layer_1], axis=3)
    convolution_layer_9 = Conv3D(64, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_9)
    convolution_layer_9 = BatchNormalization()(convolution_layer_9)
    convolution_layer_9 = Activation('relu')(convolution_layer_9)
    convolution_layer_9 = Conv3D(64, (3, 3, 3), padding='same', kernel_initializer='he_normal')(convolution_layer_9)
    convolution_layer_9 = BatchNormalization()(convolution_layer_9)
    convolution_layer_9 = Activation('relu')(convolution_layer_9)

    convolution_layer_9_shape_3d = convolution_layer_9.shape
    convolution_layer_9 = Reshape((convolution_layer_9_shape_3d[1], convolution_layer_9_shape_3d[2], convolution_layer_9_shape_3d[3] * convolution_layer_9_shape_3d[4]))(convolution_layer_9)
    output_layer = Conv2D(self.utils.n_features, (1, 1), activation='softmax', name='output_layer')(convolution_layer_9)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=20000, decay_rate=0.99)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate_scheduler), loss={'output_layer': 'categorical_crossentropy'}, metrics=[tf.keras.metrics.MeanIoU(num_classes = self.utils.n_features)])
    return model

  def train(self):
    '''
    This method internally train the U-Net defined above, with a given set of hyperparameters.
    '''
    if self.utils.pre_load_dataset == True:
      print("Loading Training & Validation Dataset...")
      self.utils.X_train = np.load('X_train.npy')
      self.utils.X_validation = np.load('X_validation.npy')
      self.utils.y_train = np.load('y_train.npy')
      self.utils.y_validation = np.load('y_validation.npy')
      print("Training and Validation Dataset Loaded")
    else:
      print("Data Loading...")
      X, y = self.utils.DataLoader(self.dataset)
      print("Data Loading Completed")
      if self.utils.layer_standardization == True:
        print("Layer-Wise Standardization...")
        X_normalized = self.utils.layer_standardization(X)
        print("Layer-Wise Standardization Completed")
      else:
        X_normalized = X
      print("Prepare Data for Training, Validation & Testing...")
      X_processed, y_processed = self.utils.prepare_dataset_for_training(X_normalized, y)
      X_train, X_, y_train, y_ = train_test_split(X_processed, y_processed, train_size = 1 - self.test_ratio, test_size = self.test_ratio, random_state=1234)
      X_validation, X_test, y_validation, y_test = train_test_split(X_, y_, train_size = 1 - self.test_ratio, test_size = self.test_ratio, random_state=1234)
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
        unet = load_model('best_model.h5', custom_objects=custom_objects)
        unet.summary()
    else:
        unet = self.build_3d_unet()
        unet.summary()
    print("Training Begins...")
    unet.fit(x = self.utils.X_train, y = self.utils.y_train, batch_size = self.utils.batch_size, epochs=self.utils.num_epochs, validation_data=(self.utils.X_validation, self.utils.y_validation), callbacks=[tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True), tf.keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)])
    print("Training Ended, Model Saved!")
    return None

  def predict(self, new_data = None):
    '''
    This method will take a pre-trained model and make corresponding predictions.
    '''
    unet = self.build_3d_unet()
    unet.load_weights('best_model.h5')
    if new_data is not None:
      n_features = self.utils.n_features
      if self.utils.layer_standardization == True:
        print("Layer-Wise Standardization...")
        X_normalized = self.utils.layer_standardization(new_data)
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
        self.utils.X_test = np.load('X_test.npy')
        self.utils.y_test = np.load('y_test.npy')
        print("Testing Dataset Loaded")
      else:
        print("Testing Begins...")
      total_test_length = (self.utils.X_test.shape[0]//self.utils.batch_size)*self.utils.batch_size
      prediction_result = np.zeros(shape=(total_test_length, self.utils.resized_x_y, self.utils.resized_x_y, self.utils.n_features))
      for i in range(0, total_test_length, self.utils.batch_size):
        print("Testing sample from:", i, "to:", i+4)
        prediction_result[i:i+4] = unet.predict(self.utils.X_test[i:i+4])

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
