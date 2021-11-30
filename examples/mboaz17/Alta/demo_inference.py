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

for i in range(10):
    n = i  # np.random.choice(len(test_dataset))

    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]

    gt_mask_pos = gt_mask.sum(axis=0)
    gt_mask = gt_mask.argmax(axis=0)

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_score = (pr_mask.squeeze().cpu().numpy()).max(axis=0)
    pr_mask = (pr_mask.squeeze(dim=0).cpu().numpy().round())
    pr_mask = pr_mask.argmax(axis=0)
    pr_mask_vis = 0*image_vis
    gt_mask_vis = 0*image_vis
    for i in range(len(train_dataset.class_values)):
        inds = (pr_mask == i).nonzero()
        pr_mask_vis[inds[0], inds[1], :] = train_dataset.class_values[i]
        inds = ((gt_mask == i) & (gt_mask_pos > 0)).nonzero()
        gt_mask_vis[inds[0], inds[1], :] = train_dataset.class_values[i]

    iou = (pr_mask_vis == gt_mask_vis).all(axis=2).mean()
    print('IOU = {}'.format(iou))
    visualize(
        image=image_vis,
        ground_truth_mask=gt_mask_vis,
        predicted_mask=pr_mask_vis,
        softmax_score=pr_score,
    )

# TODO:
# 1) Support class_intervals for batch size > 1
# 2) Support class_intervals for DiceLoss?