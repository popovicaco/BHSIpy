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
import matplotlib.pyplot as plt
import scipy.io
import os
import spectral

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
  def __init__(self, dataset = 'Pavia_University', test_ratio = 0.7, window_size = 25, num_epochs = 1000, num_components_to_keep = 10, reduced_size_x_y = 320, n_features = 5):
    self.dataset = dataset
    self.test_ratio = test_ratio
    self.window_size = window_size
    self.num_epochs = num_epochs
    self.num_components_to_keep = num_components_to_keep
    self.reduced_size_x_y = reduced_size_x_y
    self.n_features = n_features

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
        Data = scipy.io.loadmat(os.path.join(data_path, 'BiomedicalDenoisedEyeData4Endmembers.mat'))['hyperspectral_image']
        Ground_Truth = scipy.io.loadmat(os.path.join(data_path, 'BiomedicalDenoisedEyeData4Endmembers.mat'))['ground_truth']
    return Data, Ground_Truth

  def DataPreprocessing(self, X, y):
    '''
    This Method internally preprocess the input dataset to 3-d Hyper U-Net acceptable format.
    '''
    n_features = self.n_features
    feature_encoded_data = np.zeros((X.shape[0], X.shape[1], n_features))
    for i, unique_value in enumerate(np.unique(y)):
        feature_encoded_data[:, :, i][y == unique_value] = 1
    X, pca = self.run_PCA(image_cube = X, num_principal_components = self.num_components_to_keep)
    X = cv2.resize(X, (self.reduced_size_x_y, self.reduced_size_x_y))
    feature_encoded_data = cv2.resize(feature_encoded_data, (self.reduced_size_x_y, self.reduced_size_x_y))
    y = cv2.resize(y, (self.reduced_size_x_y, self.reduced_size_x_y))
    y = feature_encoded_data.reshape(1, self.reduced_size_x_y, self.reduced_size_x_y, n_features)
    X_train = X.reshape(-1, self.reduced_size_x_y, self.reduced_size_x_y, self.num_components_to_keep, 1)
    return X_train, y

  def Train_Test_Split(self, X, y, test_ratio = 0.7, randomState=1234):
    '''
    Train Test Split. 
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, random_state = randomState, stratify = y)
    return X_train, X_test, y_train, y_test

  def run_PCA(self, image_cube, num_principal_components = 30):
    '''
    Apply Principal Component Analysis to decompose the amount of features w.r.t their orthogonality, Default keeping 30 features. 
    '''
    new_cube = np.reshape(image_cube, (-1, image_cube.shape[2]))
    pca = PCA(n_components = num_principal_components, whiten=True)
    new_cube = pca.fit_transform(new_cube)
    new_cube = np.reshape(new_cube, (image_cube.shape[0], image_cube.shape[1], num_principal_components))
    return new_cube, pca

  def padding(self, image_cube, margin=2):
    '''
    Padding Function
    '''
    new_cube = np.zeros((image_cube.shape[0] + 2 * margin, image_cube.shape[1] + 2* margin, image_cube.shape[2]))
    x_margin = margin
    y_margin = margin
    new_cube[x_margin:image_cube.shape[0] + x_margin, y_margin:image_cube.shape[1] + y_margin, :] = image_cube
    return new_cube

  def create_image_cubes(self, image_cube, ground_truth, window_size = 5, remove_unecessary_labels = True):
    '''
    Create image cubes for training and testing.
    '''
    end_index = int((window_size - 1) / 2)
    padded_image_cube = self.padding(image_cube, margin = end_index)
    # split batches
    data_batch = np.zeros((image_cube.shape[0] * image_cube.shape[1], window_size, window_size, image_cube.shape[2]))
    batch_labels = np.zeros((image_cube.shape[0] * image_cube.shape[1]))
    index = 0
    for i in range(end_index, padded_image_cube.shape[0] - end_index):
        for j in range(end_index, padded_image_cube.shape[1] - end_index):
            batch = padded_image_cube[(i-end_index):(i+end_index+1), (j-end_index):(j+end_index+1)]   
            data_batch[index, :, :, :] = batch
            batch_labels[index] = y[(i-end_index), (j-end_index)]
            index = index + 1
    if remove_unecessary_labels:
        data_batch = data_batch[batch_labels > 0, :, :, :]
        batch_labels = batch_labels[batch_labels > 0]
        batch_labels -= 1
    return data_batch, batch_labels

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
    convolution_layer_5 = tf.reshape(convolution_layer_5, [1, int((self.reduced_size_x_y / 16)**2), 512, 1])
    pooling_layer_5 = MaxPooling2D(pool_size=(2, 2), padding='same')(convolution_layer_5)
    
    convolution_layer_6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(convolution_layer_5)
    convolution_layer_6 = Conv2D(1024, (3, 3), activation='relu', padding='same')(convolution_layer_6)
    
    flatten_layer_6 = Flatten()(convolution_layer_4)
    output_layer_1 = Dense(9, activation='softmax', name='output_layer_1')(flatten_layer_6)
    
    up_layer_7 = concatenate([Conv2DTranspose(512, (2, 2), padding='same')(convolution_layer_6), convolution_layer_5], axis=3)
    convolution_layer_7 = Conv2D(512, (3, 3), activation='relu', padding='same')(up_layer_7)
    convolution_layer_7 = Conv2D(512, (3, 3), activation='relu', padding='same')(convolution_layer_7)
    
    convolution_layer_5 = tf.reshape(convolution_layer_5, [1, int(self.reduced_size_x_y / 16), int(self.reduced_size_x_y / 16), 1, 512])
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

    up_layer_10 = concatenate([Conv3DTranspose(64, (2, 2,2 ), strides=(2, 2, 2), padding='same')(convolution_layer_9), convolution_layer_2], axis=3)
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
    X, y = self.DataLoader(self.dataset)
    X_train, y = self.DataPreprocessing(X, y)
    unet = self.build_3d_unet()
    unet.summary()
    unet.fit(x = X_train, y = y, batch_size = 1, epochs=self.num_epochs)
    unet.save('U-Net.hdf5')
    return None
  
  def predict(self):
    '''
    This method will take a pre-trained model and make corresponding predictions.
    '''
    X, y = self.DataLoader(self.dataset)
    X_test, y = self.DataPreprocessing(X, y)
    unet = load_model('U-Net.hdf5')
    prediction_result = unet.predict(X_test)

    prediction_encoded = np.zeros((self.reduced_size_x_y, self.reduced_size_x_y))
    for i in range(self.reduced_size_x_y):
      for j in range(self.reduced_size_x_y):
        index = np.argmax(prediction_result[0][i][j])
        if prediction_result[0][i][j][index] < 0.85:
          prediction_encoded[i][j] = 0
        else:
          prediction_encoded[i][j] = np.argmax(prediction_result[0][i][j])
    X, ground_truth = self.DataLoader(self.dataset)
    prediction = cv2.resize(prediction_encoded, (X.shape[1], X.shape[0]))
    return prediction, ground_truth

  def evaluation_metrics(self, ground_truth, prediction):
    '''
    This function returns some evaluation metrics on the test/train dataset.
    '''
    intersect = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    iou = np.sum(intersect) / np.sum(union)
    accuracy = np.sum((prediction == ground_truth).astype(int))/(prediction.shape[0]*prediction.shape[1])
    print("Accuracy: ", accuracy, " Test IOU Score: ", iou)
    return iou, accuracy
