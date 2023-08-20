from utils import utils
from unet_3d import UNet
from unet_3d_plus_plus import UNet_plus_plus
from absl import app
from absl import flags

flags.DEFINE_string('method', 'unet', 'deep learning method (Default 3d-unet)')
flags.DEFINE_string('dataset', 'biomedical_image_2023', 'training dataset (Default summer 2023 FUSRP eye image dataset (dimension 210*210))')
flags.DEFINE_integer('train_sim', 200, 'training epochs (Default 200 epochs)')
flags.DEFINE_float('test_ratio', 0.3, 'train test split ratio (Default 30%)')
flags.DEFINE_integer('batch_size', 4, 'batch_size (Default 4)')
flags.DEFINE_integer('resized_x_y', 256, 'resize original HSI image x,y dimension (Default 256*256 pixels)')
flags.DEFINE_integer('num_features', 4, 'number of features to be classified (Default 4)')
flags.DEFINE_integer('num_masks', 10, 'number of masks applied on each augmented image (Default 10)')
flags.DEFINE_integer('mask_length', 205, 'mask length of each squared mask (Default 205)')
flags.DEFINE_integer('num_components_to_keep', 3, 'number of principle component to keep (Default 3)')
flags.DEFINE_boolean('pre_load_dataset', False, 'If use pre_loaded dataset (Default False)')
flags.DEFINE_boolean('layer_standardization', True, 'If use layer_standardization (Default True)')
flags.DEFINE_boolean('eval_only', False, 'If eval_only (Default False)')
flags.DEFINE_integer('max_pca_iterations', 30, 'maximum number of PCA iterations (Default 30)')
flags.DEFINE_boolean('svd_denoising', False, 'If use svd denoise the raw datacube (Default False)')
flags.DEFINE_integer('n_svd', 4, 'initial number of SVD components (Default 4)')
flags.DEFINE_float('svd_denoise_threshold', 0.999, 'SVD Denoise Variance Threshold (Default 99.9%)')
flags.DEFINE_boolean('continue_training', False, 'If continue training from where it is left off (Default False)')
flags.DEFINE_float('dropout_rate', 0.5, 'dropout rate (Default 50%)')
flags.DEFINE_float('PCA_variance_threshold', 0.995, 'PCA Variance Threshold (Default 99.5%)')
flags.DEFINE_boolean('deep_supervision', True, 'If using deep supervision for UNet++ & 3+ (Default True)')


FLAGS = flags.FLAGS

def main(argv):
    Utils = utils(dataset = FLAGS.dataset, test_ratio = FLAGS.test_ratio, num_epochs=FLAGS.train_sim, resized_x_y=FLAGS.resized_x_y, n_features=FLAGS.num_features, num_masks = FLAGS.num_masks, mask_length = FLAGS.mask_length, batch_size = FLAGS.batch_size, pre_load_dataset = FLAGS.pre_load_dataset, layer_standardization = FLAGS.layer_standardization, num_components_to_keep = FLAGS.num_components_to_keep, max_pca_iterations = FLAGS.max_pca_iterations, svd_denoising = FLAGS.svd_denoising, n_svd=FLAGS.n_svd, svd_denoise_threshold=FLAGS.svd_denoise_threshold, continue_training = FLAGS.continue_training, dropout_rate = FLAGS.dropout_rate, PCA_variance_threshold = FLAGS.PCA_variance_threshold, deep_supervision = FLAGS.deep_supervision)
    if FLAGS.method == "unet":
        Model_UNet = UNet(utils = Utils)
    elif FLAGS.method == "unet_plus_plus":
        Model_UNet = UNet_plus_plus(utils = Utils)
    
    if FLAGS.eval_only:
        prediction_encoded, y_test = Model_UNet.predict()
        accuracy, precision, recall, f1_score, iou = Model_UNet.utils.evaluation_metrics(prediction_encoded, y_test)
    else:
        Model_UNet.train()
        Model_UNet.utils.pre_load_dataset = True
        prediction_encoded, y_test = Model_UNet.predict()
        accuracy, precision, recall, f1_score, iou = Model_UNet.utils.evaluation_metrics(prediction_encoded, y_test)
        
if __name__ == '__main__':
    app.run(main)
