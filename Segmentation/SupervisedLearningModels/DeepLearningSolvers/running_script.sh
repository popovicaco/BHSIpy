# 3D-U-Net
python3 run.py -method=unet -dataset=biomedical_image_2023 -train_sim=50 -batch_size=8 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-CBAM-U-Net
python3 run.py -method=cbam_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=4 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-U-Net++
python3 run.py -method=unet_plus_plus -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -eval_only=True

# 3D-U-Net 3+
python3 run.py -method=unet_3plus -dataset=biomedical_image_2023 -train_sim=200 -batch_size=4 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# V-Net
python3 run.py -method=vnet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -eval_only=True

#-------------------------------------------------------done--------------------------------------------------------------

# 3D-Inception U-Net
python3 run.py -method=inception_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -dropout_rate=0.1 -override_trained_optimizer=False

# 3D-Residual U-Net
python3 run.py -method=residual_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=4 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -eval_only=True

# 3D-Residual U-Net++
python3 run.py -method=residual_unet_plus_plus -dataset=biomedical_image_2023 -train_sim=200 -batch_size=1 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-Dense U-Net
python3 run.py -method=dense_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -eval_only=True

# 3D-R2U-Net
python3 run.py -method=r2unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=4 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -r2unet_recur_num=3 -eval_only=True

# 3D-R2U-Net++
python3 run.py -method=r2unet_plus_plus -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -r2unet_recur_num=3

# 3D-Attention-U-Net
python3 run.py -method=attention_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-CBAM-Attention-U-Net
python3 run.py -method=cbam_attention_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=4 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-Attention-U-Net++
python3 run.py -method=attention_unet_plus_plus -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-Attention-U-Net 3+
python3 run.py -method=attention_unet_3plus -dataset=biomedical_image_2023 -train_sim=200 -batch_size=4 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-Residual-Attention-U-Net
python3 run.py -method=residual_attention_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -eval_only=True

# 3D-Residual-Attention-U-Net++
python3 run.py -method=residual_attention_unet_plus_plus -dataset=biomedical_image_2023 -train_sim=200 -batch_size=1 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-Residual-Attention-U-Net 3+
python3 run.py -method=residual_attention_unet_3plus -dataset=biomedical_image_2023 -train_sim=200 -batch_size=4 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-R2AU-Net
python3 run.py -method=r2aunet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=4 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -r2unet_recur_num=3

# Focus UNet
python3 run.py -method=focus_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -eval_only=True


#### --------------- For IJCAI 2025 -------------
# Our Model (Done Training Sanity Check)
python3 run.py -method=r2a2_3d_unet_plus_plus -dataset=biomedical_image_2023 -train_sim=20 -batch_size=1 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -r2unet_recur_num=1 -r2unet_stacks=1 -eval_only=False

# 3D-U-Net (Done Training Sanity Check)
python3 run.py -method=unet -dataset=biomedical_image_2023 -train_sim=100 -batch_size=16 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-U-Net++ (Done Training Sanity Check)
python3 run.py -method=unet_plus_plus -dataset=biomedical_image_2023 -train_sim=50 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# V-Net (Done Training Sanity Check)
python3 run.py -method=vnet -dataset=biomedical_image_2023 -train_sim=50 -batch_size=1 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-Dense U-Net (Done Training Sanity Check)
python3 run.py -method=dense_unet -dataset=biomedical_image_2023 -train_sim=50 -batch_size=1 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-Residual U-Net (Done Training Sanity Check)
python3 run.py -method=residual_unet -dataset=biomedical_image_2023 -train_sim=50 -batch_size=1 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-R2U-Net (Done Training Sanity Check)
python3 run.py -method=r2unet -dataset=biomedical_image_2023 -train_sim=50 -batch_size=1 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -r2unet_recur_num=3

# 3D-Attention-U-Net (Done Training Sanity Check)
python3 run.py -method=attention_unet -dataset=biomedical_image_2023 -train_sim=50 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 3D-Residual-Attention-U-Net (Done Training Sanity Check)
python3 run.py -method=residual_attention_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -eval_only=True

# Focus UNet (Done Training Sanity Check)
python3 run.py -method=focus_unet -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -eval_only=True

# 2D-U-Net (Done Training Sanity Check)
python3 run.py -method=2d_unet -dataset=biomedical_image_2023 -train_sim=100 -batch_size=16 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False

# 2D-U-Net++ (Done Training Sanity Check)
python3 run.py -method=unet_2d_plus_plus -dataset=biomedical_image_2023 -train_sim=200 -batch_size=2 -pre_load_dataset=True -num_masks=10 -mask_length=205 -layer_standardization=True -num_components_to_keep=4 -num_features=4 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False




python3 run.py -method=r2a2_3d_unet_plus_plus -dataset=usable_eye_image_2 -train_sim=20 -batch_size=1 -pre_load_dataset=True -num_masks=5 -mask_length=195 -layer_standardization=False -num_components_to_keep=4 -num_features=9 -resized_x_y=256 -PCA_variance_threshold=0.995 -continue_training=False -r2unet_recur_num=1 -r2unet_stacks=1 -eval_only=False
