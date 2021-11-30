import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from datasets import Dataset
from augmentations import get_training_augmentation, get_validation_augmentation, get_preprocessing

## Create model and train
import torch
import numpy as np
import segmentation_models_pytorch as smp

# %%
# images_dir = '/media/isl12/Alta/V7_Exp_25_1_21'
# annotations_dir = '/media/isl12/Alta/V7_Exp_25_1_21_annot'
# dataset_name = 'Agamim/Path/A/30'

images_dir = '/home/airsim/repos/segmentation_models.pytorch/examples/data/Alta/train'
annotations_dir = '/home/airsim/repos/segmentation_models.pytorch/examples/data/Alta/trainannot'
dataset_name = ''

x_train_dir = os.path.join(images_dir, dataset_name)
y_train_dir = os.path.join(annotations_dir, dataset_name)
x_valid_dir = os.path.join(images_dir, dataset_name)
y_valid_dir = os.path.join(annotations_dir, dataset_name)
x_test_dir = os.path.join(images_dir, dataset_name)
y_test_dir = os.path.join(annotations_dir, dataset_name)

# %%
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

augmented_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    classes=['transportation terrain'],
)

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = [cls for ind, cls in enumerate(augmented_dataset.CLASS_NAMES) if ind>=-1]  # ind in [1, 6, 8, 10, 11, 12, 13]]
ACTIVATION = 'softmax2d'  # 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

# load best saved checkpoint
best_model = torch.load('./best_model.pth')

# %%

# create test dataset
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)

# %%

## Visualize predictions
# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir,
    classes=CLASSES,
)

# %%

# dataset_mode = 'from_dataset'
dataset_mode = 'from_folder'
## KITTI
# images_folder = '/home/airsim/repos/segmentation_models.pytorch/examples/data/Kitti/2011_09_26_drive_0001_extract/image_02/data'
# images_folder = '/home/airsim/repos/segmentation_models.pytorch/examples/data/Kitti/2011_09_26_drive_0009_extract/image_02/data'
## AIRSIM
images_folder = '/home/airsim/repos/segmentation_models.pytorch/examples/data/Airsim/train'
## ALTA
# images_folder = '/home/airsim/repos/segmentation_models.pytorch/examples/data/Alta/train'
# images_folder = '/media/isl12/Alta/V7_Exp_25_1_21/Agamim/Path/B/100'

for i in range(20):

    if dataset_mode == 'from_dataset':
        n = np.random.choice(len(test_dataset))
        image_vis = test_dataset_vis[n][0].astype('uint8')
        shape_orig = image_vis.shape[:2]
        gt_mask_vis = test_dataset_vis[n][1].astype('uint8')
        image, gt_mask = test_dataset[n]
        gt_mask_vis = gt_mask_vis[:, :, :1].squeeze()
    elif dataset_mode == 'from_folder':
        images_list = os.listdir(images_folder)
        n = np.random.choice(len(images_list))
        image_vis = cv2.imread(os.path.join(images_folder, images_list[n]))
        shape_orig = image_vis.shape[:2]
        # Apply the same steps on the image as for an image from the dataset
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
        if 'Airsim' in images_folder or 'Alta' in images_folder:
            image = get_augmentation_for_Airsim()(image=image_vis)['image']
        else:
            image = get_augmentation_for_kitti()(image=image_vis)['image']
        image = get_preprocessing(preprocessing_fn)(image=image)['image']

    shape_new = image.shape[1:]
    h_pad = int((shape_new[0] - shape_orig[0])/2)
    v_pad = int((shape_new[1] - shape_orig[1])/2)

    if dataset_mode == 'from_dataset':
        gt_mask = gt_mask.squeeze()
        gt_mask_pos = gt_mask.sum(axis=0)
        max_vals = gt_mask.max(axis=0)
        gt_mask = gt_mask.argmax(axis=0)
    else:
        gt_mask = np.zeros((image.shape[1], image.shape[2]))
        gt_mask_pos = gt_mask
        max_vals = np.zeros((image.shape[1], image.shape[2]))

    gt_mask = gt_mask[h_pad:shape_new[0]-h_pad, v_pad:shape_new[1]-v_pad]
    gt_mask_pos = gt_mask_pos[h_pad:shape_new[0]-h_pad, v_pad:shape_new[1]-v_pad]
    max_vals = max_vals[h_pad:shape_new[0]-h_pad, v_pad:shape_new[1]-v_pad]
    untagged_indices = (max_vals == 0).nonzero()


    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_score = (pr_mask.squeeze().cpu().numpy()).max(axis=0)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round()).argmax(axis=0)
    pr_mask = pr_mask[h_pad:shape_new[0]-h_pad, v_pad:shape_new[1]-v_pad]
    pr_mask_vis = 0*image_vis
    gt_mask_vis = 0*image_vis

    for i in range(len(train_dataset.class_values)):
        inds = (pr_mask == i).nonzero()
        pr_mask_vis[inds[0], inds[1], :] = train_dataset.class_values[i]
        inds = ((gt_mask == i) & (gt_mask_pos > 0)).nonzero()
        gt_mask_vis[inds[0], inds[1], :] = train_dataset.class_values[i]
    gt_mask_vis[untagged_indices[0], untagged_indices[1], :] = 0

    # cv2.imwrite(os.path.join(save_dir, os.path.split(image_path)[1]).replace('.JPG', '.tif'),
    #             cv2.cvtColor(pr_mask_vis, cv2.COLOR_RGB2BGR))

    visualize(
        image=image_vis,
        ground_truth_mask=gt_mask_vis,
        predicted_mask=pr_mask_vis,
        softmax_score=pr_score,  # np.percentile(pr_score, 50),
    )
    # visualize(
    #     image=image_vis,
    #     ground_truth_mask=gt_mask_vis,
    #     predicted_mask=pr_mask_vis,
    #     softmax_score=pr_score>0.95,  # np.percentile(pr_score, 50),
    #     score_map=score_map.max(axis=0) > 0.015,  # np.percentile(score_map, 50),
    # )

# TODO: for imporiving confidnece scores
# * Estimate one histogram model per class?
# * Estimate many histogram models, using clustering?
# * Multi-dim histogram models (2 or 3)
# * Determine importance of each feature level
# * Determine number of bins per feature level
# * Weigh each image using the number of contributing pixels (not equally)
# * Determine best histogram matching criterion
# * Take features before the nonlinearity (but it's more technically complicated)
# * Calculate true prob density after dim reduction - perhaps using PCA