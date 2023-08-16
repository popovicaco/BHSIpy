from re import I
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import numpy as np
import scipy.io
from scipy.ndimage import rotate

class utils:
  """
  Class for data generation, dataset loading, and other utility functions
  Important Input:  test_ratio: train_test split ratio
                    window_size: training window size
                    num_epochs: number of epochs for training
                    num_components_to_keep: number of principal components to keep after PCA
                    resized_x_y: used to rescale the image to a square image, for training purposes
                    n_features: number of features (i.e, muscle)
                    batch_size: batch size for training
                    mask_length: length of the mask applied on the augmented dataset
                    num_masks: number of random masks on each augmented data
                    pre_load_dataset: if previously genereated data for training and testing
                    layer_standardization: if layer-wise standardize the dataset
                    max_pca_iterations: maximum number of PCA iterations to keep a threshold (e.g, 99.5%) amount of variance
                    PCA_variance_threshold: the threshold amount of variance being kept after PCA
                    dropout_rate: dropout rate used in Bayesian MC dropout layers
                    continue_training: training from where it left off
  """
  def __init__(self, dataset = 'biomedical_image_2023', test_ratio = 0.3, num_epochs = 200, num_components_to_keep = 3, resized_x_y = 256, n_features = 4, batch_size = 4, mask_length = 205, num_masks = 10, pre_load_dataset = False, layer_standardization = True, max_pca_iterations = 30, PCA_variance_threshold = 0.995, dropout_rate = 0.5, continue_training = False):
    self.dataset = dataset
    self.test_ratio = test_ratio
    self.num_epochs = num_epochs
    self.num_components_to_keep = num_components_to_keep
    self.resized_x_y = resized_x_y
    self.n_features = n_features
    self.batch_size = batch_size
    self.mask_length = mask_length
    self.num_masks = num_masks
    self.pre_load_dataset = pre_load_dataset
    self.layer_standardization = layer_standardization
    self.max_pca_iterations = max_pca_iterations
    self.PCA_variance_threshold = PCA_variance_threshold
    self.dropout_rate = dropout_rate
    self.continue_training = continue_training
    self.X_train = None
    self.X_validation = None
    self.X_test = None
    self.y_train = None
    self.y_validation = None
    self.y_test = None

  def DataLoader(self, dataset):
    '''
    This function will load the selected dataset and return the original HSI image along with the ground truth label, if any.
    '''
    data_path = os.path.join(os.getcwd(), '')
    if dataset == 'pavia_university':
        Data = scipy.io.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        Ground_Truth = scipy.io.loadmat(os.path.join(data_path, 'PaviaU_GroundTruth.mat'))['paviaU_gt']
    elif dataset == 'biomedical_image_2022':
        Data = scipy.io.loadmat(os.path.join(data_path, 'Dataset/BiomedicalDenoisedEyeData4Endmembers.mat'))['hyperspectral_image']
        Ground_Truth = scipy.io.loadmat(os.path.join(data_path, 'Dataset/BiomedicalDenoisedEyeData4Endmembers.mat'))['ground_truth']
    elif dataset == 'biomedical_image_2023':
        Data = scipy.io.loadmat(os.path.join(data_path, 'Dataset/eye1.mat'))['Datacube'].astype(np.float64)
        Ground_Truth = np.load(os.path.join(data_path, 'Dataset/new_ground_truth.npy')).astype(np.uint8)
    return Data, Ground_Truth

  def layer_standardization(self, hyperspectral_image):
    '''
    This method performs layer-wise standardization on the input data.
    '''
    epsilon = 1e-7  # To prevent division by zero
    normalized_HSI_cube = np.zeros(hyperspectral_image.shape)
    for k in range(hyperspectral_image.shape[2]):
        x_k_mean = np.mean(hyperspectral_image[:,:,k])
        x_k_std = np.std(hyperspectral_image[:,:,k])
        normalized_HSI_cube[:,:,k] = (hyperspectral_image[:,:,k] - x_k_mean) / (x_k_std + epsilon)
    return normalized_HSI_cube

  def run_PCA(self, image_cube, num_principal_components = 3):
    '''
    Apply principal component analysis to decompose the amount of features w.r.t their orthogonality, default keeping 3 principal components.
    '''
    new_cube = np.reshape(image_cube, (-1, image_cube.shape[2]))
    pca = PCA(n_components = num_principal_components, whiten=True)
    new_cube = pca.fit_transform(new_cube)
    new_cube = np.reshape(new_cube, (image_cube.shape[0], image_cube.shape[1], num_principal_components))
    return new_cube, pca

  def DataPreprocessing(self, X, y):
    '''
    This Method internally preprocess the input dataset (and its ground truths) to its PCA reduced format.
    '''
    n_features = self.n_features
    X_, pca = self.run_PCA(image_cube = X, num_principal_components = self.num_components_to_keep)
    total_variance_explained = sum(pca.explained_variance_ratio_)
    for i in range(self.max_pca_iterations):
        if total_variance_explained <= self.PCA_variance_threshold:
            self.num_components_to_keep += 1
            X_, pca = self.run_PCA(image_cube = X, num_principal_components = self.num_components_to_keep)
            total_variance_explained = sum(pca.explained_variance_ratio_)
        else:
            break
    X_after_preprocessed = cv2.resize(X_, (self.resized_x_y, self.resized_x_y), interpolation = cv2.INTER_LANCZOS4)
    ground_truth_after_preprocessed = cv2.resize(y, (self.resized_x_y, self.resized_x_y), interpolation = cv2.INTER_NEAREST)
    return X_after_preprocessed, ground_truth_after_preprocessed

  def Spatial_Transform_Data_Augmentation(self, original_hyperspectral_image, original_hyperspectral_image_segmentation_labels, mask_length = 205, num_masks = 10):
    """
    This method applies spatial transform to augment training data.
    """
    augmented_data = [original_hyperspectral_image]
    augmented_labels = [original_hyperspectral_image_segmentation_labels]
    rotation_angles = list(range(0, 360, 15)) # Rotate 15 degrees each
    for i in range(len(rotation_angles)):
      rotated_hyperspectral_image = rotate(original_hyperspectral_image, angle=rotation_angles[i], axes=(0, 1), reshape=False, mode = "mirror", order = 0)
      left_right_flipped_image = np.fliplr(rotated_hyperspectral_image)  # horizontal flip
      up_down_flipped_image = np.flipud(rotated_hyperspectral_image)  # vertical flip
      augmented_data.append(rotated_hyperspectral_image)
      augmented_data.append(left_right_flipped_image)
      augmented_data.append(up_down_flipped_image)
      rotated_hyperspectral_image_labels = rotate(original_hyperspectral_image_segmentation_labels, angle=rotation_angles[i], axes=(0, 1), reshape=False, mode = "mirror", order = 0)
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
        x_random = np.random.randint(low = 0, high = width - mask_length)
        y_random = np.random.randint(low = 0, high = height - mask_length)
        augmented_masked_data.append(augmented_data[i][x_random:(x_random + mask_length), y_random:(y_random + mask_length), :])
        augmented_masked_labels.append(augmented_labels[i][x_random:(x_random + mask_length), y_random:(y_random + mask_length)])
    return np.array(augmented_masked_data, dtype=float), np.array(augmented_masked_labels, dtype=float)

  def DataProcessing(self, X, y):
    '''
    This Method internally preprocess the input dataset to its 3-D Hyper UNet accepted format.
    '''
    n_features = self.n_features
    feature_encoded_data = np.zeros((X.shape[0], X.shape[1], n_features))
    for i, unique_value in enumerate(np.unique(y)):
        feature_encoded_data[:, :, i][y == unique_value] = 1
    feature_encoded_data = cv2.resize(feature_encoded_data, (self.resized_x_y, self.resized_x_y), interpolation = cv2.INTER_NEAREST)
    y_after_processed = feature_encoded_data.reshape(1, self.resized_x_y, self.resized_x_y, n_features)
    X_after_processed = X.reshape(-1, self.resized_x_y, self.resized_x_y, self.num_components_to_keep, 1)
    return X_after_processed, y_after_processed

  def prepare_dataset_for_training(self, X, y):
    '''
    This method prepares dataset for training the 3-D unet in its batch_size format.
    '''
    X_augmented, y_augmented = self.Spatial_Transform_Data_Augmentation(X, y, num_masks = self.num_masks, mask_length = self.mask_length)
    X_after_preprocessed, ground_truth_after_preprocessed = self.DataPreprocessing(X_augmented[0], y_augmented[0])
    print("Number of principal components kept: ", self.num_components_to_keep)
    X_after_processed, y_after_processed = self.DataProcessing(X_after_preprocessed, ground_truth_after_preprocessed)
    for i in range(1, X_augmented.shape[0]):
      X_, y_ = self.DataPreprocessing(X_augmented[i], y_augmented[i])
      X_, y_ = self.DataProcessing(X_, y_)
      X_after_processed = np.concatenate((X_after_processed, X_), axis=0)
      y_after_processed = np.concatenate((y_after_processed, y_), axis=0)

    return X_after_processed, y_after_processed

  def evaluation_metrics(self, prediction, ground_truth):
    '''
    This function returns some useful evaluation metrics on the test/train dataset.
    '''
    intersect = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    accuracy = np.sum((prediction == ground_truth).astype(int))/(prediction.shape[0]*prediction.shape[1]*prediction.shape[2])
    true_positive = np.sum(intersect)
    false_negative = np.sum(ground_truth) - true_positive
    false_positive = np.sum(prediction) - true_positive
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)
    iou = np.sum(intersect) / np.sum(union)
    print("Accuracy: ", accuracy, " Precision: ", precision, " Recall: ", recall, " F-1 Score: ", f1_score, " Test IOU Score: ", iou)
    return accuracy, precision, recall, f1_score, iou