import os
import cv2
import tensorflow as tf
import keras
from keras.utils import np_utils
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Conv3D, Reshape, concatenate, Conv2D, MaxPooling3D, Conv3DTranspose, Flatten, Dense, BatchNormalization, MaxPooling2D, Conv2DTranspose
from keras.layers.core import Dropout
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
from plotly.offline import init_notebook_mode
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io
from scipy.ndimage import rotate
import os

class U_Net:
  """
  Class for Deep Learning Hyperspectral Segmentation with U-Net
  Input:  dataset: Name of the Dataset
          test_ratio: train_test ratio
          window_size: training window size
          num_epochs: number of epochs to train
          num_components_to_keep: number of bands to keep
          reduced_size_x_y: used to rescale the image to a square image, for training purposes
  """
  def __init__(self, dataset = 'Pavia_University', test_ratio = 0.7, window_size = 25, num_epochs = 1000, num_components_to_keep = 10, reduced_size_x_y = 320, n_features = 5, batch_size = 128, mask_length = 25, num_masks = 10, pre_load_dataset = False, layer_normalization = True):
    self.dataset = dataset
    self.test_ratio = test_ratio
    self.window_size = window_size
    self.num_epochs = num_epochs
    self.num_components_to_keep = num_components_to_keep
    self.reduced_size_x_y = reduced_size_x_y
    self.n_features = n_features
    self.batch_size = batch_size
    self.mask_length = mask_length
    self.num_masks = num_masks
    self.pre_load_dataset = pre_load_dataset
    self.X_train = None
    self.X_validation = None
    self.X_test = None
    self.y_train = None
    self.y_validation = None
    self.y_test = None
    self.layer_normalization = layer_normalization

  def DataLoader(self, dataset):
    '''
    This function will load the selected dataset and return the processed dataset for training and testing along with a ground truth label.
    right now only Pavia University dataset is being used for testing purposes.
    '''
    data_path = os.path.join(os.getcwd(), '')
    if dataset == 'Pavia_University':
        Data = scipy.io.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        Ground_Truth = scipy.io.loadmat(os.path.join(data_path, 'PaviaU_GroundTruth.mat'))['paviaU_gt']
    elif dataset == 'biomedical_image':
        Data = scipy.io.loadmat(os.path.join(data_path, 'Dataset/BiomedicalDenoisedEyeData4Endmembers.mat'))['hyperspectral_image']
        Ground_Truth = scipy.io.loadmat(os.path.join(data_path, 'Dataset/BiomedicalDenoisedEyeData4Endmembers.mat'))['ground_truth']
    return Data, Ground_Truth
  
  def Layer_Normalization(self, hyperspectral_image):
    '''
    This method performs layer-wise normalization on the input data.
    '''
    normalized_HSI_cube = np.zeros(hyperspectral_image.shape)
    for i in range(hyperspectral_image.shape[0]):
      for j in range(hyperspectral_image.shape[1]):
        x_k_min = np.min(hyperspectral_image[i,j,:])
        x_k_max = np.max(hyperspectral_image[i,j,:])
        for k in range(hyperspectral_image.shape[2]):
          normalized_HSI_cube[i,j,k] = (hyperspectral_image[i,j,k] - x_k_min)/(x_k_max - x_k_min)
    return normalized_HSI_cube

  def Spatial_Transform_Data_Augmentation(self, original_hyperspectral_image, original_hyperspectral_image_segmentation_labels, mask_length = 25, num_masks = 10):
    """
    This method applied spatial transform to augment training data.
    """
    augmented_data = []
    augmented_labels = []
    rotation_angles = list(range(0, 360, 15)) # Rotate 15 degrees each
    for i in range(len(rotation_angles)):
      rotated_hyperspectral_image = rotate(original_hyperspectral_image, angle=rotation_angles[i], axes=(0, 1), reshape=False, mode = "grid-constant", order = 0)  # rotates the image by 45 degrees
      left_right_flipped_image = np.fliplr(rotated_hyperspectral_image)  # horizontal flip
      up_down_flipped_image = np.flipud(rotated_hyperspectral_image)  # vertical flip
      augmented_data.append(rotated_hyperspectral_image)
      augmented_data.append(left_right_flipped_image)
      augmented_data.append(up_down_flipped_image)
      rotated_hyperspectral_image_labels = rotate(original_hyperspectral_image_segmentation_labels, angle=rotation_angles[i], axes=(0, 1), reshape=False, mode = "grid-constant", order = 0)  # rotates the labels by 45 degrees
      left_right_flipped_image = np.fliplr(rotated_hyperspectral_image_labels)  # horizontal flip
      up_down_flipped_image = np.flipud(rotated_hyperspectral_image_labels)  # vertical flip
      augmented_labels.append(rotated_hyperspectral_image_labels)
      augmented_labels.append(left_right_flipped_image)
      augmented_labels.append(up_down_flipped_image)

    augmented_masked_data = []
    augmented_masked_labels = []
    # Create Masked Datasets
    for i in range(len(augmented_data)):
      height, width, channels = augmented_data[i].shape
      height_labels, width_labels = augmented_labels[i].shape
      for j in range(num_masks):
        new_mask = np.zeros((height, width), dtype=bool)
        x_random = np.random.randint(low = 0, high = width - mask_length)
        y_random = np.random.randint(low = 0, high = height - mask_length)
        new_mask[x_random:x_random + mask_length, y_random:y_random + mask_length] = True
        masked_image = np.ma.masked_array(augmented_data[i], mask=np.repeat(new_mask[:, :, np.newaxis], channels, axis=2))
        augmented_masked_data.append(masked_image.mask*augmented_data[i])
        augmented_masked_labels.append(new_mask*augmented_labels[i])
    return np.array(augmented_masked_data, dtype=object), np.array(augmented_masked_labels, dtype=object)

  def DataPreprocessing(self, X, y):
    '''
    This Method internally preprocess the input dataset to 3-d Hyper U-Net acceptable format.
    '''
    n_features = self.n_features
    num_iterations = X.shape[0]
    feature_encoded_data = np.zeros((X[0].shape[0], X[0].shape[1], n_features))
    for i, unique_value in enumerate(np.unique(y[0])):
        feature_encoded_data[:, :, i][y[0] == unique_value] = 1
    X_, pca = self.run_PCA(image_cube = X[0], num_principal_components = self.num_components_to_keep)
    X_ = cv2.resize(X_, (self.reduced_size_x_y, self.reduced_size_x_y), interpolation = cv2.INTER_AREA)
    X_after_processed = X_.reshape(-1, self.reduced_size_x_y, self.reduced_size_x_y, self.num_components_to_keep, 1)
    feature_encoded_data = cv2.resize(feature_encoded_data, (self.reduced_size_x_y, self.reduced_size_x_y), interpolation = cv2.INTER_AREA)
    y_after_processed = feature_encoded_data.reshape(1, self.reduced_size_x_y, self.reduced_size_x_y, n_features)
    for j in range(1, num_iterations):
      feature_encoded_data = np.zeros((X[j].shape[0], X[j].shape[1], n_features))
      for k, unique_value in enumerate(np.unique(y[j])):
          feature_encoded_data[:, :, k][y[j] == unique_value] = 1
      X_pca, pca = self.run_PCA(image_cube = X[j], num_principal_components = self.num_components_to_keep)
      X_pca = cv2.resize(X_pca, (self.reduced_size_x_y, self.reduced_size_x_y), interpolation = cv2.INTER_AREA)
      X_pca = X_pca.reshape(-1, self.reduced_size_x_y, self.reduced_size_x_y, self.num_components_to_keep, 1)
      feature_encoded_data = cv2.resize(feature_encoded_data, (self.reduced_size_x_y, self.reduced_size_x_y), interpolation = cv2.INTER_AREA)
      y_ = feature_encoded_data.reshape(1, self.reduced_size_x_y, self.reduced_size_x_y, n_features)
      X_after_processed = np.concatenate((X_after_processed, X_pca), axis = 0)
      y_after_processed = np.concatenate((y_after_processed, y_), axis = 0)
    return X_after_processed, y_after_processed

  def run_PCA(self, image_cube, num_principal_components = 30):
    '''
    Apply Principal Component Analysis to decompose the amount of features w.r.t their orthogonality, Default keeping 30 features.
    '''
    new_cube = np.reshape(image_cube, (-1, image_cube.shape[2]))
    pca = PCA(n_components = num_principal_components, whiten=True)
    new_cube = pca.fit_transform(new_cube)
    new_cube = np.reshape(new_cube, (image_cube.shape[0], image_cube.shape[1], num_principal_components))
    return new_cube, pca

  def build_3d_unet(self):
    '''
    This function is a tensorflow realization of the 3-d hyper U-Net model proposed by Nishchal et. al 2021.
    Manuscript Name: Pansharpening and Semantic Segmentation of Satellite Imagery.
    Link: https://ieeexplore.ieee.org/document/9544725
    '''
    input_layer = Input((self.reduced_size_x_y, self.reduced_size_x_y, self.num_components_to_keep, 1))
    convolution_layer_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(input_layer)
    convolution_layer_1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(convolution_layer_1)
    convolution_layer_1 = BatchNormalization()(convolution_layer_1)
    pooling_layer_1 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(convolution_layer_1)

    convolution_layer_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pooling_layer_1)
    convolution_layer_2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(convolution_layer_2)
    convolution_layer_2 = BatchNormalization()(convolution_layer_2)
    convolution_layer_2 = Dropout(0.1)(convolution_layer_2)
    pooling_layer_2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(convolution_layer_2)

    convolution_layer_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pooling_layer_2)
    convolution_layer_3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(convolution_layer_3)
    convolution_layer_3 = BatchNormalization()(convolution_layer_3)
    convolution_layer_3 = Dropout(0.1)(convolution_layer_3)
    pooling_layer_3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(convolution_layer_3)

    convolution_layer_4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pooling_layer_3)
    convolution_layer_4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(convolution_layer_4)
    convolution_layer_4 = BatchNormalization()(convolution_layer_4)
    convolution_layer_4 = Dropout(0.1)(convolution_layer_4)
    pooling_layer_4 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(convolution_layer_4)

    convolution_layer_5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pooling_layer_4)
    convolution_layer_5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(convolution_layer_5)
    convolution_layer_5 = BatchNormalization()(convolution_layer_5)
    convolution_layer_5 = Dropout(0.2)(convolution_layer_5)
    convolution_layer_5 = tf.reshape(convolution_layer_5, [1, convolution_layer_5.shape[1]*convolution_layer_5.shape[2], 512, 1])
    pooling_layer_5 = MaxPooling2D(pool_size=(2, 2), padding='same')(convolution_layer_5)

    convolution_layer_6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(convolution_layer_5)
    convolution_layer_6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(convolution_layer_6)

    flatten_layer_6 = Flatten()(convolution_layer_4)
    output_layer_1 = Dense(9, activation='softmax', name='output_layer_1')(flatten_layer_6)

    up_layer_7 = concatenate([Conv2DTranspose(512, (2, 2), padding='same')(convolution_layer_6), convolution_layer_5], axis=3)
    convolution_layer_7 = Conv2D(512, (3, 3), activation='relu', padding='same')(up_layer_7)
    convolution_layer_7 = Conv2D(512, (3, 3), activation='relu', padding='same')(convolution_layer_7)

    convolution_layer_5 = tf.reshape(convolution_layer_5, [1, int(math.sqrt(convolution_layer_5.shape[1])), int(math.sqrt(convolution_layer_5.shape[1])), 1, 512])
    up_layer_8 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(convolution_layer_5), convolution_layer_4], axis=3)
    convolution_layer_8 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up_layer_8)
    convolution_layer_8 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(convolution_layer_8)
    convolution_layer_8 = BatchNormalization()(convolution_layer_8)
    convolution_layer_8 = Dropout(0.2)(convolution_layer_8)

    up_layer_9 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(convolution_layer_4), convolution_layer_3], axis=3)
    convolution_layer_9 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up_layer_9)
    convolution_layer_9 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(convolution_layer_9)
    convolution_layer_9 = BatchNormalization()(convolution_layer_9)
    convolution_layer_9 = Dropout(0.1)(convolution_layer_9)
    print(convolution_layer_9.shape)

    up_layer_10 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(convolution_layer_9), convolution_layer_2], axis=3)
    convolution_layer_10 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up_layer_10)
    convolution_layer_10 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(convolution_layer_10)
    convolution_layer_10 = BatchNormalization()(convolution_layer_10)
    convolution_layer_10 = Dropout(0.1)(convolution_layer_10)

    up_layer_11 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2,2, 2), padding='same')(convolution_layer_10), convolution_layer_1], axis=3)
    convolution_layer_11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up_layer_11)
    convolution_layer_11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(convolution_layer_11)

    layer_shape_3d = convolution_layer_11.shape
    convolution_layer_11 = Reshape((layer_shape_3d[1], layer_shape_3d[2], layer_shape_3d[3] * layer_shape_3d[4]))(convolution_layer_11)
    output_layer_2 = Conv2D(self.n_features, (1, 1), activation='softmax', name='output_layer_2')(convolution_layer_11)
    model = Model(inputs=[input_layer], outputs=[output_layer_2])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss={'output_layer_2': 'categorical_crossentropy'}, metrics=[tf.keras.metrics.MeanIoU(num_classes = self.n_features)])
    return model

  def train(self):
    '''
    This method internally train the U-Net defined above, with a given set of hyperparameters.
    '''
    if self.pre_load_dataset == True:
      print("Loading Training & Validation Dataset...")
      self.X_train = np.load('X_train.npy')
      self.X_validation = np.load('X_validation.npy')
      self.y_train = np.load('y_train.npy')
      self.y_validation = np.load('y_validation.npy')
      print("Training and Validation Dataset Loaded")
    else:
      print("Data Loading...")
      X, y = self.DataLoader(self.dataset)
      print("Data Loading Completed")
      if self.layer_normalization == True:
        print("Layer-Wise Normalization...")
        X_normalized = self.Layer_Normalization(X)
        print("Layer-Wise Normalization Completed")
      print("Data Augmentation...")
      X_augmented, y_augmented = self.Spatial_Transform_Data_Augmentation(X_normalized, y, num_masks = self.num_masks)
      print("Data Augmentation Completed")
      print("Data Processing...")
      X_processed, y_processed = self.DataPreprocessing(X_augmented, y_augmented)
      X_train, X_, y_train, y_ = train_test_split(X_processed, y_processed, train_size = 0.7, test_size = 0.3, random_state=1234)
      X_validation, X_test, y_validation, y_test = train_test_split(X_, y_, train_size = 0.7, test_size = 0.3, random_state=1234)
      self.X_train, self.X_validation, self.X_test = X_train, X_validation, X_test
      self.y_train, self.y_validation, self.y_test = y_train, y_validation, y_test
      np.save('X_train.npy', self.X_train)
      np.save('X_validation.npy', self.X_validation)
      np.save('X_test.npy', self.X_test)
      np.save('y_train.npy', self.y_train)
      np.save('y_validation.npy', self.y_validation)
      np.save('y_test.npy', self.y_test)
      print("Data Processing Completed")
    unet = self.build_3d_unet()
    unet.summary()
    print("Training Begins...")
    unet.fit(x = self.X_train, y = self.y_train, batch_size = self.batch_size, epochs=self.num_epochs, validation_data=(self.X_validation, self.y_validation))
    unet.save('U-Net.hdf5')
    print("Training Ended, Model Saved!")
    return None

  def predict(self, new_data = None):
    '''
    This method will take a pre-trained model and make corresponding predictions.
    '''
    unet = load_model('U-Net.hdf5')
    if new_data is not None:
      n_features = self.n_features
      if self.layer_normalization == True:
        print("Layer-Wise Normalization...")
        X_normalized = self.Layer_Normalization(new_data)
        print("Layer-Wise Normalization Completed")
      X_, pca = self.run_PCA(image_cube = X_normalized, num_principal_components = self.num_components_to_keep)
      X_ = cv2.resize(X_, (self.reduced_size_x_y, self.reduced_size_x_y), interpolation = cv2.INTER_AREA)
      X_test = X_.reshape(-1, self.reduced_size_x_y, self.reduced_size_x_y, self.num_components_to_keep, 1)
      prediction_result = unet.predict(X_test)
      prediction_encoded = np.zeros((self.reduced_size_x_y, self.reduced_size_x_y))
      for i in range(self.reduced_size_x_y):
        for j in range(self.reduced_size_x_y):
          index = np.argmax(prediction_result[0][i][j])
          if prediction_result[0][i][j][index] < 0.85:
            prediction_encoded[i][j] = 0
          else:
            prediction_encoded[i][j] = np.argmax(prediction_result[0][i][j])
      prediction = cv2.resize(prediction_encoded, (new_data.shape[1], new_data.shape[0]), interpolation = cv2.INTER_AREA)
      return prediction
    else:
      if self.pre_load_dataset == True:
        print("Loading Testing Dataset...")
        self.X_test = np.load('X_test.npy')
        self.y_test = np.load('y_test.npy')
        print("Testing Dataset Loaded")
      else:
        print("Data Loading...")
        X, y = self.DataLoader(self.dataset)
        print("Data Loading Completed")
        if self.layer_normalization == True:
          print("Layer-Wise Normalization...")
          X_normalized = self.Layer_Normalization(X)
          print("Layer-Wise Normalization Completed")
        print("Data Augmentation...")
        X_augmented, y_augmented = self.Spatial_Transform_Data_Augmentation(X, y, num_masks = self.num_masks)
        print("Data Augmentation Completed")
        print("Data Processing...")
        X_processed, y_processed = self.DataPreprocessing(X_augmented, y_augmented)
        X_train, X_, y_train, y_ = train_test_split(X_processed, y_processed, train_size = 0.7, test_size = 0.3, random_state=1234)
        X_validation, X_test, y_validation, y_test = train_test_split(X_, y_, train_size = 0.7, test_size = 0.3, random_state=1234)
        self.X_train, self.X_validation, self.X_test = X_train, X_validation, X_test
        self.y_train, self.y_validation, self.y_test = y_train, y_validation, y_test
        print("Data Processing Completed")
      prediction_result = unet.predict(self.X_test)

      prediction_ = np.zeros(shape=(self.X_test.shape[0], self.reduced_size_x_y, self.reduced_size_x_y))
      for k in range(self.X_test.shape[0]):
        prediction_encoded = np.zeros((self.reduced_size_x_y, self.reduced_size_x_y))
        for i in range(self.reduced_size_x_y):
          for j in range(self.reduced_size_x_y):
            index = np.argmax(prediction_result[k][i][j])
            if prediction_result[k][i][j][index] < 0.85:
              prediction_encoded[i][j] = 0
            else:
              prediction_encoded[i][j] = np.argmax(prediction_result[k][i][j])
        prediction = cv2.resize(prediction_encoded, (self.X_test[k].shape[1], self.X_test[k].shape[0]), interpolation = cv2.INTER_AREA)
        prediction_[k] = prediction
      return prediction_, np.argmax(self.y_test, axis=-1)

  def evaluation_metrics(self, ground_truth, prediction):
    '''
    This function returns some evaluation metrics on the test/train dataset.
    '''
    intersect = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    iou = np.sum(intersect) / np.sum(union)
    accuracy = np.sum((prediction == ground_truth).astype(int))/(prediction.shape[0]*prediction.shape[1]*prediction.shape[2])
    print("Accuracy: ", accuracy, " Test IOU Score: ", iou)
    return iou, accuracy
